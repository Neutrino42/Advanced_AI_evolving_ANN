import numpy as np
import random
import copy

random.seed('EPNet')

m = 5  # input nodes number

n = 1  # output nodes number, by default, this variable will not be used.
       # Modifying it changes nothing

MAX_HID_NODES = 5
density = 0.75
epoch = 500


class Network:

    def __init__(self, max_hid_nodes, edge_dens):

        N = max_hid_nodes

        self.dim = m + N + 1
        self.connect_mat = np.zeros([self.dim, self.dim]) # 1 or 0, size: (m + N + n)*(m + N + n), lower triagular
        self.weight_mat = np.random.standard_normal([self.dim, self.dim]) # real number, size: (m + N + n)*(m + N + n), lower triagular
        self.hidden_nodes = [] # 1 or 0, size: 1*N
        self.node_num = m + 1
        self.X = np.zeros(self.dim)
        self.net = np.zeros(self.dim)
        # Feedback of net, which is the gradient of Loss function respect to self.net
        self.F_net = np.zeros(self.dim)
        self.F_weight = np.zeros([self.dim, self.dim])
        self.test = np.zeros([self.dim, self.dim])  # the importance for each connection

        # randomly generate hidden nodes
        # the valid hidden nodes will always be at the beginning of the list
        tmp = random.randint(1, N)
        self.node_num += tmp
        self.hidden_nodes = [1 for i in range(tmp)]
        self.hidden_nodes += [0 for i in range(N-tmp)]

        edge_num = (self.node_num*(self.node_num - 1)/2 - (m*(m-1)/2)) * edge_dens
        count = 0

        # Add connections for input and output nodes to other nodes.
        # Add to output node first
        node_idx = random.randint(0, self.dim - 2)
        self.connect_mat[self.dim-1][node_idx] = 1
        self.connect_mat[node_idx][self.dim-1] = 1  ######################
        self.weight_mat[self.dim-1][node_idx] = random.random() / 4  ######################
        self.weight_mat[node_idx][self.dim-1] = self.weight_mat[self.dim-1][node_idx]
        count += 1
        # Then add to input nodes
        for i in range(m):

            is_added = False

            while not is_added:
                # randomly pick a node
                node_idx = random.randint(m, self.dim-1)
                # Check its existence, especially when it may be a hidden node
                if node_idx != self.dim-1:
                    if self.hidden_nodes[node_idx - m] == 0:
                        continue

                # Check if the connection already exists
                if self.connect_mat[i][node_idx] == 1:
                    continue

                # Update connection matrix and weight matrix
                self.connect_mat[i][node_idx] = 1
                self.connect_mat[node_idx][i] = 1 ######################
                self.weight_mat[i][node_idx] = random.random()/4 ######################
                self.weight_mat[node_idx][i] = self.weight_mat[i][node_idx]

                count += 1
                is_added = True
        #print(self.connect_mat)

        # 接着加新的edges
        while count < edge_num:
            is_added = False

            while not is_added:
                sample_list = [i for i in range(self.node_num -1)] + [self.dim-1]
                idx_1, idx_2 = random.sample(sample_list, 2)
                # Check their existence, especially when they may be a hidden node
                if idx_1 == idx_2:
                    continue
                if idx_1 < m and idx_2 < m:
                    continue
                if m <= idx_1 < self.dim - 1:
                    if self.hidden_nodes[idx_1 - m] == 0:
                        continue
                if m <= idx_2 < self.dim - 1:
                    if self.hidden_nodes[idx_2 - m] == 0:
                        continue

                # Check if the connection already exists
                if self.connect_mat[idx_1][idx_2] == 1:
                    continue

                # Update connection matrix and weight matrix
                self.connect_mat[idx_1][idx_2] = 1
                self.connect_mat[idx_2][idx_1] = 1  ######################
                self.weight_mat[idx_1][idx_2] = random.random() / 4  ######################
                self.weight_mat[idx_2][idx_1] = self.weight_mat[idx_1][idx_2]
                count += 1
                is_added = True

        # lower-triagularize the two matrices
        self.connect_mat = np.tril(self.connect_mat, k=-1)
        self.weight_mat = np.tril(self.weight_mat, k=-1)
        #print(self.connect_mat)



    # By default, the network's output is one node
    # return: scalar
    # 可以改成输入多个test case，用矩阵运算
    def feedforward(self, input_set):
        self.X[:len(input_set)] = input_set

        for i in range(m, self.dim):
            self.net[i] = np.dot((self.connect_mat[i]*self.weight_mat[i]), self.X) #######
            self.X[i] = sigmoid(self.net[i])
        return self.X[-1]


    def calc_error(self, test_set, desired_out):
        """

        :param test_set: the validation test suite grouped in 2-d list
        :param desired_out: the desired output for each test, in 1-d list
        :return: the overall error
        """
        tmp_out = []
        for i in test_set:
            tmp_out.append(self.feedforward(i))
        tmp_out = np.array(tmp_out)
        E = 100 * ( ((tmp_out-np.array(desired_out))**2).sum() )/(len(desired_out)*1)
        return E #################????


    def get_answers(self, test_set):
        tmp_out = []
        for i in test_set:
            tmp_out.append(self.feedforward(i))
        return np.array(tmp_out)


    def back_prop(self, input_set, desired_output):
        """
        This BP is calculated using one training sample
        :param input_set:
        :param desired_output:
        :return: F_weight(input_set), gradient of Loss function respect to training sample input_set
        """
        y_hat = self.feedforward(input_set) # user output
        F_y_hat = np.zeros(self.dim)
        F_y_hat[-1] = y_hat - desired_output  # [0, 0, 0, ... , 0, y_hat]
        # filter only the valid nodes
        s_prime_net = sigmoid_prime(self.net) * ([0 for k in range(m)] + self.hidden_nodes + [1])
        for i in range(self.dim-1, m-1, -1): # from self.dim-1 to m (boundary included)
            #print(sigmoid_prime(self.net))
            self.F_net = s_prime_net * (F_y_hat + np.dot((self.connect_mat * self.weight_mat).transpose(), self.F_net))
        a = np.reshape(self.F_net, [len(self.F_net), 1])  # dim * 1 vector
        b = np.reshape(self.X, [1, len(self.X)])          # 1 * dim vector
        gradient = np.tril(np.dot(a, b), k=-1)
        return gradient


    def train(self, training_set, output_set, learning_rate):  # learning_rate > 0
        weight_history = np.zeros([len(training_set), self.dim, self.dim])

        for x, y, i in zip(training_set, output_set, range(len(training_set))):
            gradient_mat = self.back_prop(x, y)
            self.weight_mat -= learning_rate * gradient_mat # update weight
            weight_history[i] = copy.deepcopy(self.weight_mat)

        weight_sum = weight_history.sum(0)
        avg = weight_sum / weight_history.shape[0]
        sqrt_sum = np.sqrt( np.square(weight_history-avg).sum(0))
        # set the zero terms to 1 to avoid division by zero
        self.test = weight_sum / np.where(sqrt_sum==0, 1, sqrt_sum)
        # ?????????

    def epoch_train(self, training_set, output_set, eta, epoch):
        error_before = self.calc_error(training_set, output_set)
        weight_copy = copy.deepcopy(self.weight_mat)

        for i in range(1, epoch+1):
            self.train(training_set, output_set, eta)
            # check error every 10 epochs
            if i % (epoch//15) == 0:
                error_after = self.calc_error(training_set, output_set)
                if error_after <= error_before:
                    eta = eta*1.2
                    error_before = error_after
                    weight_copy = copy.deepcopy(self.weight_mat)
                else:
                    eta = eta*0.8
                    # restore the previous weight
                    self.weight_mat = copy.deepcopy(weight_copy)
        return self.calc_error(training_set, output_set)


    def SA_train(self, test_set, output_set):
        T = 3000
        epo = 200
        while T > 10:
            for i in range(epo):
                weight_copy = copy.deepcopy(self.weight_mat)

                old_error = self.calc_error(test_set, output_set)

                perturb = np.random.standard_normal([self.dim, self.dim])
                perturb = np.tril(perturb, k=-1)
                self.weight_mat += perturb

                new_error = self.calc_error(test_set, output_set)
                delta_E = new_error - old_error
                if delta_E < 0:
                    pass
                else:
                    if random.random() < np.exp(-delta_E/T):
                        pass
                    else:
                        self.weight_mat = weight_copy
            T = 0.9*T
        return self.calc_error(test_set, output_set)

    def calc_approx_impt(self):
        pass

    def add_connection(self, max_num):
        filter_mat = np.tril(np.ones([self.dim, self.dim]), k=-1) - self.connect_mat
        arg_sorted_mat = (filter_mat * self.test).flatten().argsort()
        for index in arg_sorted_mat[-random.randint(1, max_num):]:
            x = index // self.dim
            y = index % self.dim
            if y > x:
                tmp = x
                x = y
                y = tmp
            self.connect_mat[x, y] = 1
            self.weight_mat[x, y] = self.test[x, y]


    def delete_conn(self, max_num):
        """
        Delete based on self.test, which records the importance of each connection
        :return:
        """
        # sort from smallest to largest
        arg_sorted_mat = (self.connect_mat * self.test).flatten().argsort()
        # selection ---------------------------- need modification
        for index in arg_sorted_mat[:random.randint(1, max_num)]:
            x = index // self.dim
            y = index % self.dim
            self.connect_mat[x,y] = 0
            self.connect_mat[y,x] = 0
            self.weight_mat[x,y] = 0
            self.weight_mat[y,x] = 0



    def delete_nodes(self, num):
        """

        :param num: max number of nodes to be deleted
        :return: number of nodes deleted
        """
        count = 0
        hid_n_size = self.node_num - m - 1 # number of hidden nodes
        for i in range(hid_n_size):
            if count == num:
                break
                # s 个里面取 num 个，每一个的概率是 p = 1-C(s-1,num)/C(s,num) = num/s,
                # 这样，每个球有p的概率被取到，取到球的总数的期望为s*p = num

                # If a node is deleted:
                # reorganize corresponding arrays to maintain the state
                # where valid nodes (represented by 1) are arranged to the head-most part of the arrays
                # e.g. hidden_nodes:
                #      before:            [1, 1,       1, 1, 0, 0]
                #      after deletion:    [1, null,    1, 1, 0, 0]
                #                             |        |  |
                #                             o<------(1  1)
                #                             |        |
                #      reorganize:        [1, 1,       1, 0, 0, 0]
            if random.random() < num/hid_n_size:
                # delete node
                self.hidden_nodes = self.hidden_nodes[1:] + [0]
                # delete connections and update matrices
                # discard row i and column i
                # Delete the i th column/row, then add one column/row of zeros before the last column/row of the original matrix

                m_i = i + m  # convert index
                #-----------update connection matrix--------------
                tmp_mat = self.connect_mat
                tmp_mat = np.delete(tmp_mat, m_i, 0)
                tmp_mat = np.insert(tmp_mat, -1, np.zeros(self.dim), 0)
                tmp_mat = np.delete(tmp_mat, m_i, 1)
                self.connect_mat = np.insert(tmp_mat, -1, np.zeros(self.dim), 1)
                self.connect_mat = np.tril(self.connect_mat, k=-1)

                # -----------update weight matrix--------------
                tmp_mat = self.weight_mat
                tmp_mat = np.delete(tmp_mat, m_i, 0)
                tmp_mat = np.insert(tmp_mat, -1, np.zeros(self.dim), 0)
                tmp_mat = np.delete(tmp_mat, m_i, 1)
                self.weight_mat = np.insert(tmp_mat, -1, np.zeros(self.dim), 1)
                self.weight_mat = np.tril(self.weight_mat, k=-1)
                self.node_num -= 1
                count += 1
        return count


    def cell_div(self, alpha):
        # add node only when there is vacant position
        if self.hidden_nodes.count(1) > 0:
            # randomly choose a node to duplicate
            while True:
                parent_index = m + random.randint(0,len(self.hidden_nodes)-1)
                if self.hidden_nodes[parent_index-m] == 1:
                    break

            # insert the new node next to its parent
            new_index = parent_index + 1

            self.hidden_nodes = [1] + self.hidden_nodes[:-1]
            self.node_num += 1

            # Update connection and weight matrices
            tmp_mat = self.connect_mat
            tmp_mat = np.delete(tmp_mat, -2, 0)
            tmp_mat = np.insert(tmp_mat, new_index, self.connect_mat[parent_index, :], 0)
            tmp_mat = np.delete(tmp_mat, -2, 1)
            self.connect_mat = np.insert(tmp_mat, new_index, self.connect_mat[:, parent_index], 1)

            #self.connect_mat[new_index, :] = self.connect_mat[parent_index, :]
            #self.connect_mat[:, new_index] = self.connect_mat[:, parent_index]
            self.connect_mat[new_index, new_index] = 0

            # 从前面连过来的weight不变，往后连的weight都要变
            tmp_mat = self.weight_mat
            tmp_mat = np.delete(tmp_mat, -2, 0)
            tmp_mat = np.insert(tmp_mat, new_index, self.weight_mat[parent_index, :], 0)
            tmp_mat = np.delete(tmp_mat, -2, 1)
            self.weight_mat = np.insert(tmp_mat, new_index, self.weight_mat[:, parent_index], 1)

            # edges from the beginning
            self.weight_mat[new_index, :parent_index+1] = -alpha * self.weight_mat[parent_index, :parent_index+1]
            # edges connecting forward
            self.weight_mat[parent_index+1:, new_index] = -alpha * self.weight_mat[parent_index+1:, parent_index]
            self.weight_mat[parent_index+1:, parent_index] = (1+alpha) * self.weight_mat[parent_index+1:, parent_index]

            # lower triagularization
            self.connect_mat = np.tril(self.connect_mat, k=-1)
            self.weight_mat = np.tril(self.weight_mat, k=-1)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    # Derivative of the sigmoid function.
    return sigmoid(z)*(1-sigmoid(z))


def population_init(M, test_set, desired_out):
    population = [Network(MAX_HID_NODES, density) for i in range(M)]
    is_success = [False for i in range(M)]
    init_error = [net.calc_error(test_set, desired_out) for net in population]
    return population, is_success, init_error


def error_sort(population, is_success, error):
    # best to worst. For error: smallest to largest
    zipped = sorted(zip(population, is_success, error), key=lambda item: item[2])
    pop, suc, e = zip(*zipped)
    return list(pop), list(suc), list(e)


def choose(sorted_popu):
    M = len(sorted_popu)
    # initialize return values
    p_idx = 0
    parent_c = sorted_popu[p_idx]
    for i in range(M):
        # probability for choosing (M-j)th individual:
        # P(M-j) = j/sum(1 to M)
        if random.random() < (M-i)/((1+M)*M/2):
            p_idx = i
            parent_c = sorted_popu[p_idx]
            break
    return parent_c, p_idx


def training_set_init(dim):
    """

    :param dim: dimension of the problem
    :return: x: all the combination. y: desired output for each case in x
    """
    rows = 2**dim
    x = np.zeros([rows, dim])
    for col in range(x.shape[1]):
        x[:, col] = [i // (2 ** (dim - col - 1)) % 2 for i in range(rows)]
    y = (x.sum(1) + 1)%2
    return x, y

def main():
    M = 20
    learning_rate = 0.3 ###########################
    training_set, output_set= training_set_init(5)
    test_set = training_set ################
    test_out = output_set ##################
    population, is_success, error = population_init(M, test_set, test_out)
    for t in range(100):
        #选parent
        population, is_success, error = error_sort(population, is_success, error)
        parent, p_index = choose(population)
        print("epoch: {}. parent @ {}".format(t,p_index))
        print(error)
        print(is_success)
        print(parent.get_answers(training_set))
        offspring = copy.deepcopy(parent)
        if is_success[p_index] == 1:
            error[p_index] = parent.epoch_train(training_set, output_set, learning_rate, epoch)
            print("success and train")
            continue
        else: # parent failure, train with SA
            error_before = error[p_index]
            offspring.SA_train( training_set, output_set)
            error_after = offspring.calc_error(test_set, test_out)
            if error_before-error_after > error_before * 0.2:
                # replace the parent
                population[p_index] = copy.deepcopy(offspring)
                error[p_index] = error_after
                is_success[p_index] = True
                print("success and SA train")
                continue
            else: # delete nodes
                offspring.delete_nodes(n) # n 不能比MAX_HID_NODE大
                error_after = offspring.epoch_train(training_set, output_set, learning_rate, epoch)
                # if better than the worst one, replace it
                if error_after < error[-1]:
                    error[-1] = error_after
                    population[-1] = copy.deepcopy(offspring)
                    print("nodes deletion")
                    continue
                else: # delete connections
                    #offspring.calc_approx_impt() #######
                    offspring.delete_conn(3)
                    ###########
                    #############
                    ###########
                    error_after = offspring.epoch_train(training_set, output_set, learning_rate, epoch)
                    # if better than the worst one, replace it
                    if error_after < error[-1]:
                        error[-1] = error_after
                        population[-1] = copy.deepcopy(offspring)
                        print("connection deletion")
                        continue
                    else: # add connections and nodes
                        offspring2 = copy.deepcopy(offspring)
                        offspring.add_connection(3)#########
                        #################
                        #################
                        offspring2.cell_div(-0.4) ##########
                        #################
                        ###############
                        ###############

                        error_after = offspring.epoch_train(training_set, output_set, learning_rate, epoch)
                        error_after_2 = offspring2.epoch_train(training_set, output_set, learning_rate, epoch)
                        # replace the worst in one the population
                        if error_after < error_after_2:
                            population[-1] = copy.deepcopy(offspring)
                            error[-1] = error_after
                            print("connecton addition")
                        else:
                            population[-1] = copy.deepcopy(offspring2)
                            error[-1] = error_after_2
                            print("nodes addition")


    output_net = population[0]
    output_net.epoch_train(training_set, output_set, learning_rate, epoch)
    E = output_net.calc_error(training_set, output_set)
    print(E)
    print(output_net.hidden_nodes)
    print(output_net.connect_mat)
    print(output_net.weight_mat)
    print(output_net.get_answers(training_set))
    print(output_set)

    #net = Network(3,0.7)
    #print(net.weight_mat)
    #print(net.connect_mat)
    #print(net.hidden_nodes)



if __name__ == '__main__':

    main()
'''
    M = 20
    learning_rate = 0.3  ###########################
    training_set, output_set = training_set_init(5)
    test_set = training_set  ################
    test_out = output_set  ##################
    population, is_success, error = population_init(M, test_set, test_out)
    parent, p_index = choose(population)
    offspring = copy.deepcopy(parent)
    offspring.cell_div(-0.4)
'''
