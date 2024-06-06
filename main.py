import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# Generate a random regular graph with degree 3 and 10 nodes
input_graph = nx.random_regular_graph(3, 1000)

recovey_rate = 1.0
infection_rate = 0.8
initial_infected = 0.1
horizon = 30.0

# Draw the graph with node labels
plt.figure(figsize=(8, 8))  # Add this line to adjust the figure size for better visibility
nx.draw(input_graph, with_labels=True, node_size=70, font_size=5, node_color='skyblue', alpha=0.8)

# Save the figure
plt.savefig("input_graph.png")
plt.close()  # Close the figure to free up memory


# init 0 is S, 1 is I
number_of_nodes = input_graph.number_of_nodes()
edge_list = [(u, v) for u, v in input_graph.edges()] + [(v, u) for u, v in input_graph.edges()] # important: each edge has to be there twice in both directions
node_values = [0 if random.random() > initial_infected else 1 for i in range(number_of_nodes)]
upper_rate_inf = len(edge_list) * infection_rate
upper_rate_rec =  number_of_nodes * recovey_rate
upper_rate = upper_rate_inf + upper_rate_rec  # this can even be optimized further
time = 0.0

# set up times
expected_residence_time = 1.0/upper_rate
expected_steps = int(horizon/expected_residence_time)
residence_times = np.random.exponential(expected_residence_time, expected_steps*2)

# Compute the cumulative sum
cumulative_residence_times = np.cumsum(residence_times)
jump_times = cumulative_residence_times[cumulative_residence_times <= horizon]
#jump_times = np.concatenate((cumulative_residence_times, [horizon]))

num_infected = np.sum(node_values)
jump_times_series = []
num_infected_series = []

# Print the first few cumulative residence times to check
print(jump_times[:10]) 
print(jump_times[-10:]) 


jump_times_record = np.linspace(0.0, horizon, 1000)
num_infected_record = list()
pointer = 0 

# algorithm
num_jumps = len(jump_times)
for i in range(num_jumps):
    current_time = jump_times[i]
    # check if infection or recovery event
    event_type = np.random.choice([0, 1], p=[upper_rate_rec/upper_rate, upper_rate_inf/upper_rate]) # 0 is recovery, 1 is infection
    if event_type == 0:
        # recovery event
        node = np.random.randint(0, number_of_nodes)
        if node_values[node] == 1:
            node_values[node] = 0
            num_infected -= 1 
            while jump_times_record[pointer] < current_time:
                num_infected_record.append(num_infected + 1)
                pointer += 1
    else:
        # infection event
        edge = random.choice(edge_list)
        if node_values[edge[0]] == 1 and node_values[edge[1]] == 0:
            node_values[edge[1]] = 1
            num_infected += 1
            while jump_times_record[pointer] < current_time:
                num_infected_record.append(num_infected - 1)
                pointer += 1


while jump_times_record[pointer] <= horizon:
    num_infected_record.append(num_infected)
    pointer += 1
    if pointer >= len(jump_times_record):
        break



import pandas as pd

# Assuming jump_times_series and num_infected_series are already defined and of equal length

# Create a pandas DataFrame
df = pd.DataFrame({
    'JumpTime': jump_times_record,
    'NumInfected': num_infected_record
})

# Save the DataFrame to a CSV file
csv_filename = 'jump_times_and_infections.csv'
df.to_csv(csv_filename, index=False)  # index=False to avoid writing row indices to the file

print(f'Data saved to {csv_filename}.')





import matplotlib.pyplot as plt

# Assuming the DataFrame df is already defined and contains 'JumpTime' and 'NumInfected' columns

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(df['JumpTime'], df['NumInfected'], marker='o', linestyle='-', color='b')  # Plot with blue line and circle markers
plt.title('Number of Infected Over Time')  # Add a title to the plot
plt.xlabel('Time')  # Label the x-axis as "Time"
plt.ylabel('Number of Infected')  # Label the y-axis as "Number of Infected"
plt.grid(True)  # Add a grid for better readability

# Save the plot as a JPG file
jpg_filename = 'infection_plot.jpg'
plt.savefig(jpg_filename, format='jpg')  # Specify the format explicitly

plt.close()  # Close the figure to free up memory

print(f'Plot saved as {jpg_filename}.')
