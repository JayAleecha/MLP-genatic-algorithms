import matplotlib.pyplot as plt
import numpy as np

# คลาสสำหรับโหลดและจัดการข้อมูล
class DataLoader:
    def __init__(self, filename):
        self.data = self.load_data(filename)
    
    def load_data(self, filename):
        data = []
        with open(filename) as f:
            for line in f:
                elements = line.strip().split(',')
                diagnosis = 1 if elements[1] == 'M' else 0
                features = [float(element) for element in elements[2:]]
                data.append([diagnosis] + features)
        return np.array(data)
    
    def get_features_labels(self):
        X = self.data[:, 1:]
        y = self.data[:, 0]
        return X, y

    def normalize(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# คลาสสำหรับจัดการ K-fold Cross Validation
class KFoldCrossValidation:
    def __init__(self, data, k=10):
        self.data = data
        self.k = k

    def split(self):
        np.random.shuffle(self.data)
        fold_size = len(self.data) // self.k

        folds = []
        for i in range(self.k):
            validation_set = self.data[i * fold_size: (i + 1) * fold_size]
            train_set = np.concatenate((self.data[:i * fold_size], self.data[(i + 1) * fold_size:]), axis=0)
            folds.append((train_set, validation_set))
        return folds

# คลาสสำหรับ Neural Network
class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

    def initialize_weights(self):
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        individual = []
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            b = np.random.randn(layer_sizes[i + 1]) * 0.1
            individual.append((W, b))
        return individual

    def forward_propagation(self, X, individual):
        activations = X
        for W, b in individual[:-1]:
            Z = np.dot(activations, W) + b
            activations = np.tanh(Z)
        
        W, b = individual[-1]
        Z = np.dot(activations, W) + b
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # For numerical stability
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)  # Softmax for output layer

    def compute_accuracy(self, X, y, individual):
        predictions = self.forward_propagation(X, individual)
        predicted_classes = np.argmax(predictions, axis=1)
        return np.mean(predicted_classes == y)

# คลาสสำหรับการทำ Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, input_size, hidden_layers, output_size, population_size=100, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mlp = MLP(input_size, hidden_layers, output_size)

    def initialize_population(self):
        return [self.mlp.initialize_weights() for _ in range(self.population_size)]

    def select_parents(self, population, fitness, num_parents):
        parents = []
        for _ in range(num_parents):
            max_fitness_idx = np.argmax(fitness)
            parents.append(population[max_fitness_idx])
            fitness[max_fitness_idx] = -1
        return parents

    def crossover(self, parents, offspring_size):
        offspring = []
        crossover_point = offspring_size[1] // 2
        for k in range(offspring_size[0]):
            parent1_idx = k % len(parents)
            parent2_idx = (k + 1) % len(parents)
            child = []
            for i in range(len(parents[0])):
                W1, b1 = parents[parent1_idx][i]
                W2, b2 = parents[parent2_idx][i]
                W_new = np.concatenate((W1[:, :crossover_point], W2[:, crossover_point:]), axis=1)
                b_new = np.concatenate((b1[:crossover_point], b2[crossover_point:]))
                child.append((W_new, b_new))
            offspring.append(child)
        return offspring

    def mutation(self, offspring):
        for individual in offspring:
            for i in range(len(individual)):
                if np.random.rand() < self.mutation_rate:
                    W, b = individual[i]
                    W += np.random.randn(*W.shape) * 0.1
                    b += np.random.randn(*b.shape) * 0.1
                    individual[i] = (W, b)
        return offspring

    def train(self, X, y):
        population = self.initialize_population()
        for generation in range(self.generations):
            fitness = np.array([self.mlp.compute_accuracy(X, y, individual) for individual in population])
            print(f"Generation {generation}: Best Fitness = {np.max(fitness)}")
            parents = self.select_parents(population, fitness, num_parents=self.population_size // 2)
            offspring_size = (self.population_size - len(parents), len(parents[0]))
            offspring = self.crossover(parents, offspring_size)
            offspring = self.mutation(offspring)
            population = parents + offspring
        final_fitness = np.array([self.mlp.compute_accuracy(X, y, individual) for individual in population])
        return population[np.argmax(final_fitness)]

# คลาสสำหรับการทดลอง
class Experiment:
    def __init__(self, filename):
        self.data_loader = DataLoader(filename)
        X, y = self.data_loader.get_features_labels()
        self.X = self.data_loader.normalize(X)
        self.y = y

    def run(self, k=10, hidden_layer_configurations=None, population_size, generations, mutation_rate):
        if hidden_layer_configurations is None:
            hidden_layer_configurations = [[40, 35, 20]]  # 3 hidden layers with specified nodes

        kfold = KFoldCrossValidation(self.data_loader.data, k)
        folds = kfold.split()

        all_fold_accuracies = []  # เก็บค่า accuracy ของแต่ละ fold
        total_conf_matrix = np.array([[0, 0], [0, 0]])  # สำหรับเก็บค่า confusion matrix รวม

        for hidden_layers in hidden_layer_configurations:
            print(f"Testing hidden layers: {hidden_layers}")
            fold_accuracies = []

            for i, (train_set, validation_set) in enumerate(folds):
                X_train, y_train = train_set[:, 1:], train_set[:, 0]
                X_val, y_val = validation_set[:, 1:], validation_set[:, 0]

                ga = GeneticAlgorithm(X_train.shape[1], hidden_layers, 2, population_size, generations, mutation_rate)
                best_model = ga.train(X_train, y_train)
                accuracy = ga.mlp.compute_accuracy(X_val, y_val, best_model)
                fold_accuracies.append(accuracy)
                print(f"Fold {i + 1}: Accuracy = {accuracy * 100}")

                # คำนวณ confusion matrix
                predictions = np.argmax(ga.mlp.forward_propagation(X_val, best_model), axis=1)
                conf_matrix = self.compute_confusion_matrix(y_val, predictions)
                total_conf_matrix += conf_matrix  # รวม confusion matrix ของแต่ละ fold เข้าด้วยกัน

            all_fold_accuracies.append(fold_accuracies)
            print(f"Average accuracy for hidden layers {hidden_layers}: {np.mean(fold_accuracies)}")

        # Plot กราฟ accuracy ของแต่ละ fold
        self.plot_fold_accuracies(all_fold_accuracies, k)

        # Plot Confusion Matrix รวม
        self.plot_confusion_matrix(total_conf_matrix)

    def compute_confusion_matrix(self, y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return np.array([[TN, FP], [FN, TP]])

    def plot_fold_accuracies(self, all_fold_accuracies, k):
        num_configs = len(all_fold_accuracies)  # จำนวนของ hidden layer configurations
        plt.figure(figsize=(15, 5))  # ตั้งขนาดหน้าต่างกราฟ

        for i, fold_accuracies in enumerate(all_fold_accuracies):
            plt.plot(range(1, k + 1), fold_accuracies, marker='o', label=f"Hidden Layer Configuration")

        plt.title("Accuracy per Fold for Each Hidden Layer Configuration")
        plt.xlabel("Fold Number")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self, conf_matrix):
        fig, ax = plt.subplots()
        cax = ax.matshow(conf_matrix, cmap='Blues')
        for (j, k), value in np.ndenumerate(conf_matrix):
            ax.text(k, j, f"{value}", ha='center', va='center', color='red')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title("Combined Confusion Matrix")
        plt.colorbar(cax)
        plt.show()
# รันการทดลอง
if __name__ == '__main__':
    experiment = Experiment('C:/Users/Admin/MLP GA/wdbc.data')
    
    hidden_layer_configs = [[40, 20]]  # 3 hidden layers with specified nodes
    population_size=50
    generations=100
    mutation_rate=0.01
    
    experiment.run(k=10, hidden_layer_configurations=hidden_layer_configs, population_size, generations, mutation_rate)
