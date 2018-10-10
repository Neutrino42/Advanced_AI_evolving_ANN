import numpy as np
import random
import copy

random.seed('EPNet')

m = 5  # input nodes number

n = 1  # output nodes number, by default, this variable will not be used.
       # Modifying it changes nothing

MAX_HID_NODES = 5
density = 0.75

class Network:

    def __init__(self, max_hid_nodes, edge_dens):

        N = max_hid_nodes

        self.dim = m + N + 1
        self.connect_mat = np.zeros([self.dim, self.dim]) # 1 or 0, size: (m + N + n)*(m + N + n)
        self.weight_mat = np.zeros([self.dim, self.dim]) # real number, size: (m + N + n)*(m + N + n)
        self.hidden_nodes = [] # 1 or 0, size: 1*N
        self.node_num = m + 1

        # randomly generate hidden nodes
        # the valid hidden nodes will always be at the beginning of the list
        tmp = random.randint(0, N)
        self.node_num += tmp
        self.hidden_nodes = [1 for i in range(tmp)]
        self.hidden_nodes += [0 for i in range(N-tmp)]
        '''
        for i in range(N):
            if random.random() < 0.5:
                self.hidden_nodes.append(0)
            else:
                self.hidden_nodes.append(1)
                self.node_num += 1
        '''
        edge_num = self.node_num*(self.node_num - 1) / 2 * edge_dens
        print(edge_num)
        count = 0

        # Add connections for input and output nodes to other nodes.
        for i in (list(range(m)) + [self.dim-1]):

            is_added = False

            while not is_added:
                # randomly pick a node
                node_idx = random.randint(0, self.dim-1)
                # Check its existence, especially when it may be a hidden node
                if m <= node_idx < self.dim-1:
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
        print(self.connect_mat)

        # 接着加新的edges
        while count < edge_num:

            is_added = False

            while not is_added:
                idx_1 = random.randint(0, self.dim - 1)
                idx_2 = random.randint(0, self.dim - 1)
                #print(idx_1, idx_2)
                # Check their existence, especially when they may be a hidden node
                if idx_1 == idx_2:
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



    # By default, the network's output is one node
    # return: scalar
    # 可以改成输入多个test case，用矩阵运算
    def feedforward(self, input_set):
        X = [i for i in input_set]  # deep copy

        for i in range(len(self.hidden_nodes)+1):
            tmp = 0
            for j in range(m+i):
                tmp += self.weight_mat[j][i] * X[i]
            X.append(sigmoid(tmp))
        return X[-1]


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
        E = 100 * ((tmp_out.max()-tmp_out.min()) * ((tmp_out-np.array(desired_out))**2).sum() )/(len(desired_out)*1)
        return E #################????


    def back_prop(self):
        pass


    def train(self, training_set):
        pass

    def SA_train(self, training_set):
        pass

    def calc_approx_impt(self):
        pass

    def add_connection(self):
        pass

    def delete_conn(self):
        pass

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

                # -----------update weight matrix--------------
                tmp_mat = self.weight_mat
                tmp_mat = np.delete(tmp_mat, m_i, 0)
                tmp_mat = np.insert(tmp_mat, -1, np.zeros(self.dim), 0)
                tmp_mat = np.delete(tmp_mat, m_i, 1)
                self.weight_mat = np.insert(tmp_mat, -1, np.zeros(self.dim), 1)

                self.node_num -= 1
                count += 1
        return count


    def cell_div(self, alpha):
        # add node only when there is vacant position
        if self.hidden_nodes.count(0) > 0:
            # randomly choose a node to duplicate
            while True:
                parent_index = m + random.randint(0,len(self.hidden_nodes))
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
            self.weight_mat[new_index, :parent_index+1] = alpha * self.weight_mat[parent_index, :parent_index+1]
            # edges connecting forward
            self.weight_mat[parent_index+1:, new_index] = alpha * self.weight_mat[parent_index+1:, parent_index]
            self.weight_mat[parent_index+1:, parent_index] = (1+alpha) * self.weight_mat[parent_index+1:, parent_index]




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
    p_index = 0
    parent = sorted_popu[p_index]
    for i in range(M):
        # probability for choosing (M-j)th individual:
        # P(M-j) = j/sum(1 to M)
        if random.random() < (M-i)/((1+M)*M/2):
            p_index = i
            parent = sorted_popu[p_index]
            break
    return parent, p_index





def main():
    M = 20
    training_set = np.zeros() ################
    population, is_success, error = population_init(M, test_set, desired_out)
    while(True):
        #选parent
        error_sort(population, is_success, error)
        parent, p_index = choose(population)
        offspring = copy.deepcopy(parent)
        if is_success[p_index] == 1:
            parent.train(training_set) ########################
            error[p_index] = parent.calc_error(test_set) ##############
            continue
        else: # parent failure, train with SA
            error_before = error[p_index]
            offspring.SA_train(training_set)
            error_after = offspring.calc_error(test_set)
            if error_before-error_after > error_before * 0.2:
                # replace the parent
                population[p_index] = offspring
                error[p_index] = error_after
                is_success[p_index] = True
                continue
            else: # delete nodes
                offspring.delete_nodes(n) # n 不能比MAX_HID_NODE大
                offspring.train(training_set)
                error_after = offspring.calc_error(test_set)
                # if better than the worst one, replace it
                if error_after < error[-1]:
                    error[-1] = error_after
                    population[-1] = offspring
                    continue
                else: # delete connections
                    offspring.calc_approx_impt() #######
                    offspring.delete_conn()
                    ###########
                    #############
                    ###########
                    offspring.train(training_set)
                    error_after = offspring.calc_error(test_set)
                    # if better than the worst one, replace it
                    if error_after < error[-1]:
                        error[-1] = error_after
                        population[-1] = offspring
                        continue
                    else: # add connections and nodes
                        offspring2 = copy.deepcopy(offspring)
                        offspring.add_connection()#########
                        #################
                        #################
                        offspring2.cell_div() ##########
                        #################
                        ###############
                        ###############
                        offspring.train(training_set)
                        offspring2.train(training_set)
                        error_after = offspring.calc_error(test_set)
                        error_after_2 = offspring2.calc_error(test_set)

                        # replace the worst in one the population
                        if error_after < error_after_2:
                            population[-1] = offspring
                            error[-1] = error_after
                        else:
                            population[-1] = offspring2
                            error[-1] = error_after_2

    output_net = population[0]
    output_net.train(training_set)



    #net = Network(3,0.7)
    #print(net.weight_mat)
    #print(net.connect_mat)
    #print(net.hidden_nodes)



if __name__ == '__main__':
    main()
