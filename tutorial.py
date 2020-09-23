import numpy as np
import torch
import torch.nn as nn
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

#we won't need this later but for now, sure why not
def create_binary_list_from_int(number):
    if number < 0 or type(number) is not int:
        raise ValueError("Only Positive integers are allowed")

    return [int(x) for x in list(bin(number))[2:]]

def generate_even_data(max_int, batch_size):
    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [create_binary_list_from_int(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_length - len(x))) + x for x in data]

    return labels, data

class Generator(nn.Module):
    def __init__(self, input_length):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(input_length, input_length)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x.to(device)))

class Discriminator(nn.Module):
    def __init__(self, input_length):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(input_length,1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x.to(device)))

def train(max_int, batch_size, training_steps):
    save_every = 100
    input_length = int(np.log2(max_int))

    generator = Generator(input_length)
    generator.to(device)
    discriminator = Discriminator(input_length)
    discriminator.to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    loss = nn.BCELoss()

    for i in range(training_steps):
        generator_optimizer.zero_grad()

        #for some reason noise is either 0 or 1... sure jan
        noise = torch.randint(0,2,size=(batch_size, input_length)).float()
        generated_data = generator(noise)

        #now for the real datas
        true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
        true_labels = torch.tensor(true_labels).float().to(device)
        true_data = torch.tensor(true_data).float()

        #true labels are just... 1s

        #train generator (NOT discriminator)
        #invert labels!
        #if disc. thinks the false things are true, then loss is small
        #if disc. thinks the false things are false, loss is high
        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        generator_optimizer.step()

        #discriminator!
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        #detatch so we don't accidentally train the generator too
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size).to(device))

        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        if i% save_every == 0:
            print("saving, generator loss: ", generator_loss, " disc loss ", discriminator_loss)
            torch.save({"epoch": i, "model_state_dict": generator.state_dict(), "optimizer_state_dict": generator_optimizer.state_dict(),
                 "loss": generator_loss}, "checkpoints/generator_checkpoint"+str(i)+".pth.tar")
            torch.save({"epoch": i, "model_state_dict": discriminator.state_dict(),
                        "optimizer_state_dict": discriminator_optimizer.state_dict(),
                        "loss": discriminator_loss}, "checkpoints/discriminator_checkpoint"+str(i)+".pth.tar")

def test():
    generator = Generator(7)
    generator.to(device)
    discriminator = Discriminator(7)
    discriminator.to(device)

    for num in ["100","200","300","400"]:
        generator_checkpoint = torch.load("checkpoints/generator_checkpoint"+num+".pth.tar")
        generator.load_state_dict(generator_checkpoint["model_state_dict"])
        discriminator_checkpoint = torch.load("checkpoints/discriminator_checkpoint"+num+".pth.tar")
        discriminator.load_state_dict(discriminator_checkpoint["model_state_dict"])

        noise = torch.randint(0, 2, size=(8, 7)).float()
        generated_data = generator(noise).cpu().detach().numpy()
        results = []
        for j in range(generated_data.shape[0]):
            rounded = np.round(generated_data[j,:])
            #print(rounded)
            result = int("".join(str(int(k)) for k in rounded), 2)
            results.append(result)
        print(num, " ", results)


if __name__ == "__main__":
    #train(128, 16, 500)
    test()




