import warnings

import numpy as np
import random
import copy
import sys

import time

m = 5 + 1  # input nodes number + bias node

n = 1  # output nodes number, by default, this variable will not be used.
# Modifying it changes nothing

MAX_HID_NODES = 5
density = 0.98
epoch = 100


class Network:

    def __init__(self, max_hid_nodes, edge_dens=1, def_nodenum=None, weight_mat=None):
        """

        :param max_hid_nodes: maximum number of hidden nodes
        :param edge_dens: density of the edges campared to full connection
        :param def_nodenum: user-defined hidden nodes number
        """

        N = max_hid_nodes

        self.dim = m + N + 1
        self.connect_mat = np.zeros([self.dim, self.dim])  # 1 or 0, size: (m + N + n)*(m + N + n), lower triagular
        self.weight_mat = np.random.standard_normal(
            [self.dim, self.dim])  # real number, size: (m + N + n)*(m + N + n), lower triagular
        self.hidden_nodes = []  # 1 or 0, size: 1*N
        self.node_num = m + 1
        self.X = np.zeros(self.dim)
        self.net = np.zeros(self.dim)
        # Feedback of net, which is the gradient of Loss function respect to self.net
        self.F_net = np.zeros(self.dim)
        self.F_weight = np.zeros([self.dim, self.dim])
        self.test = np.zeros([self.dim, self.dim])  # the importance for each connection
        self.max_conn = 0
        self.min_conn = 0
        # randomly generate hidden nodes
        # the valid hidden nodes will always be at the beginning of the list
        if def_nodenum:
            tmp = def_nodenum
        else:
            tmp = random.randint(1, N)

        if weight_mat is not None:
            self.weight_mat = copy.deepcopy(weight_mat) + np.random.standard_normal([self.dim, self.dim]) / 6

        self.node_num += tmp
        self.hidden_nodes = [1 for i in range(tmp)]
        self.hidden_nodes += [0 for i in range(N - tmp)]

        # BY DEFAULT, FULL CONNECTION IS CONSIDERED
        if edge_dens == 1:
            self.connect_mat = np.ones([self.dim, self.dim])
            self.connect_mat[:m, :] = 0
            self.connect_mat[m + tmp:-1, :] = self.connect_mat[:, m + tmp:-1] = 0
            # lower-triagularize the two matrices
            self.connect_mat = np.tril(self.connect_mat, k=-1)
            self.weight_mat = self.weight_mat * self.connect_mat

            self.max_conn = self.connect_mat.sum()
            self.min_conn = m - 1 + self.hidden_nodes.count(1)

        else:
            count = 0
            # add connection between bias node to hidden nodes and output node
            self.connect_mat[-1, 0] = self.connect_mat[m:tmp + m, 0] = 1
            edge_num = (self.node_num * (self.node_num - 1) / 2 - (m * (m - 1) / 2)) * edge_dens
            # Add connections for input and output nodes to other nodes.
            # Add connections to output node first
            node_idx = random.randint(1, self.dim - 2)
            self.connect_mat[self.dim - 1][node_idx] = 1
            self.connect_mat[node_idx][self.dim - 1] = 1  ######################
            # self.weight_mat[self.dim-1][node_idx] = random.random() / 4  ######################
            # self.weight_mat[node_idx][self.dim-1] = self.weight_mat[self.dim-1][node_idx]
            count += 1
            # Then add connections to input nodes
            for i in range(m):

                is_added = False

                while not is_added:
                    # randomly pick a node
                    node_idx = random.randint(m, self.dim - 1)
                    # Check its existence, especially when it may be a hidden node
                    if node_idx != self.dim - 1:
                        if self.hidden_nodes[node_idx - m] == 0:
                            continue

                    # Check if the connection already exists
                    if self.connect_mat[i][node_idx] == 1:
                        continue

                    # Update connection matrix and weight matrix
                    self.connect_mat[i][node_idx] = 1
                    self.connect_mat[node_idx][i] = 1  ######################
                    # self.weight_mat[i][node_idx] = random.random()/4 ######################
                    # self.weight_mat[node_idx][i] = self.weight_mat[i][node_idx]

                    count += 1
                    is_added = True
            # print(self.connect_mat)

            # 接着加新的edges
            while count < edge_num:
                is_added = False

                while not is_added:
                    sample_list = [i for i in range(self.node_num - 1)] + [self.dim - 1]
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
                    # self.weight_mat[idx_1][idx_2] = random.random() / 4  ######################
                    # self.weight_mat[idx_2][idx_1] = self.weight_mat[idx_1][idx_2]
                    count += 1
                    is_added = True

            # lower-triagularize the two matrices
            self.connect_mat = np.tril(self.connect_mat, k=-1)
            self.weight_mat = self.weight_mat * self.connect_mat
            self.min_conn = m - 1 + self.hidden_nodes.count(1)
            self.max_conn = (m + m + self.hidden_nodes.count(1) - 1) * self.hidden_nodes.count(
                1) / 2 + m + self.hidden_nodes.count(1)

    #            print(self.connect_mat)

    # By default, the network's output is one node
    # return: scalar
    # 可以改成输入多个test case，用矩阵运算
    def feedforward(self, input_set):
        # init bias node to 1
        self.X[0] = 1
        self.X[1:1 + len(input_set)] = input_set

        for i in range(m, self.dim):
            self.net[i] = np.dot((self.connect_mat[i] * self.weight_mat[i]), self.X)  #######
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
        E = 100 * (((tmp_out - np.array(desired_out)) ** 2).sum()) / (len(desired_out) * 1)
        return E

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
        y_hat = self.feedforward(input_set)  # user output
        F_y_hat = np.zeros(self.dim)
        F_y_hat[-1] = y_hat - desired_output  # [0, 0, 0, ... , 0, y_hat]
        # filter only the valid nodes
        s_prime_net = sigmoid_prime(self.net) * ([0 for k in range(m)] + self.hidden_nodes + [1])
        for i in range(self.dim - 1, m - 1, -1):  # from self.dim-1 to m (boundary included)
            self.F_net = s_prime_net * (F_y_hat + np.dot((self.connect_mat * self.weight_mat).transpose(), self.F_net))
        a = np.reshape(self.F_net, [len(self.F_net), 1])  # dim * 1 vector
        b = np.reshape(self.X, [1, len(self.X)])  # 1 * dim vector
        gradient = np.tril(np.dot(a, b), k=-1)
        return gradient

    def train(self, training_set, output_set, learning_rate):  # learning_rate > 0
        weight_history = np.zeros([len(training_set), self.dim, self.dim])

        l_rate_mat = np.random.random([self.dim, self.dim]) * learning_rate / 10 + learning_rate - learning_rate / 20

        for x, y, i in zip(training_set, output_set, range(len(training_set))):
            gradient_mat = self.back_prop(x, y)
            self.weight_mat -= l_rate_mat * gradient_mat  # update weight
            weight_history[i] = copy.deepcopy(self.weight_mat)

        weight_sum = weight_history.sum(0)

        avg = weight_sum / weight_history.shape[0]
        sqrt_sum = np.sqrt(np.square(weight_history - avg).sum(0))
        # set the zero terms to 1 to avoid division by zero
        self.test = weight_sum / np.where(sqrt_sum == 0, 1, sqrt_sum)
        self.test[:m] = 0
        # ?????????

    def epoch_train(self, training_set, output_set, eta, epoch):
        error_before = self.calc_error(training_set, output_set)
        weight_copy = copy.deepcopy(self.weight_mat)

        # Shuffle the training set
        index = np.arange(output_set.shape[0])
        np.random.shuffle(index)
        training_set = training_set[index]
        output_set = output_set[index]

        for i in range(1, epoch + 1):
            self.train(training_set, output_set, eta)
            # check error every epoch//num epochs
            if i % (10) == 0:
                error_after = self.calc_error(training_set, output_set)
                if error_after <= error_before:
                    error_before = error_after
                    weight_copy = copy.deepcopy(self.weight_mat)
                    if eta < 0.6:
                        eta = eta * 1.1

                else:
                    if eta > 0.1:
                        eta = eta * 0.9
                        # restore the previous weight
                        self.weight_mat = copy.deepcopy(weight_copy)

        return self.calc_error(training_set, output_set)

    def SA_train(self, test_set, output_set):
        T = 3000
        epo = 50
        count = 0
        # best_error = self.calc_error(test_set, output_set)
        # best_solution = copy.deepcopy(self.weight_mat)
        while T > 5:
            sample_i = random.randint(0, len(output_set) - 1)
            # gradient_mat = self.back_prop(test_set[sample_i], output_set[sample_i])
            for i in range(epo):
                weight_copy = copy.deepcopy(self.weight_mat)

                old_error = self.calc_error(test_set, output_set)

                perturb = np.random.standard_normal([self.dim, self.dim]) / 3
                perturb = np.tril(perturb, k=-1)
                self.weight_mat += perturb

                new_error = self.calc_error(test_set, output_set)
                delta_E = new_error - old_error

                # if new_error < best_error:
                #    best_error = new_error
                #    best_solution = copy.deepcopy(self.weight_mat)

                if delta_E < 0:
                    pass
                else:
                    if random.random() < np.exp(-delta_E / T):
                        pass
                    else:
                        self.weight_mat = weight_copy
            T = 0.92 * T
            count += epo
        # elf.weight_mat = copy.deepcopy(best_solution)
        return self.calc_error(test_set, output_set), count

    def add_connection(self, max_num):

        if self.connect_mat.sum() == self.max_conn:
            return -1
        filter_mat = np.tril(np.ones([self.dim, self.dim]), k=-1) - self.connect_mat
        filter_mat[:m] = 0
        filter_mat[m + self.hidden_nodes.count(1):-1, :] = filter_mat[:, m + self.hidden_nodes.count(1):-1] = 0

        #        print(self.connect_mat)
        # base_index = filter_mat.size - np.count_nonzero(filter_mat)
        # M = np.count_nonzero(filter_mat)
        # sort from smallest to largest
        # [0 0 0 0 0 0 ... 0 0 x x x ... x x]
        #                      |->base index
        # arg_sorted_mat = (filter_mat * self.test).flatten().argsort()

        # only the available edges is sorted and stored in sorted_mat, from largest to smallest
        sorted_mat = np.sort(np.abs(self.test[filter_mat == 1]))[::-1]
        count = 0
        # for index in arg_sorted_mat[: base_index-1 : -1]:
        for edge in sorted_mat[:random.randint(1, max_num)]:

            if self.connect_mat.sum() == self.max_conn:
                #                print(self.connect_mat)
                return count

            if count == max_num:
                #                print(self.connect_mat)
                return count
            # if random.random() < index - base_index / ((1 + M) * M / 2):
            # if random.random() < index - base_index / ((1 + M) * M / 2):
            tmp = np.abs(self.test * filter_mat)
            arg_list = np.argwhere(tmp == edge)
            x, y = arg_list[0]
            if y > x:
                tmp = x
                x = y
                y = tmp
            # assert np.abs(self.test[x, y])==edge
            self.connect_mat[x, y] = 1
            self.weight_mat[x, y] = self.test[x, y]
            count += 1
        #        print(self.connect_mat)
        return count

    def delete_conn(self, max_num):
        """
        Delete based on self.test, which records the importance of each connection
        :return:
        """

        if self.connect_mat.sum() <= self.min_conn:
            return -1

        # sort from smallest to largest based on the importance of each connection
        sorted_mat = np.sort(np.abs(self.test[self.connect_mat == 1]))
        # arg_sorted_mat = (self.connect_mat * self.test).flatten().argsort()
        # selection ---------------------------- need modification
        base_index = np.count_nonzero(self.connect_mat)
        count = 0
        for edge in sorted_mat[:random.randint(1, max_num)]:
            # for index in arg_sorted_mat[base_index: base_index + random.randint(1, max_num)]:
            # M = self.dim - base_index - 1
            # for index in arg_sorted_mat[base_index: ]:

            if self.connect_mat.sum() <= self.min_conn:
                return count

            if count == max_num:
                return count
            # if random.random() < M-(index-base_index)/((1+M)*M/2):
            tmp = np.abs(self.test * self.connect_mat)
            arg_list = np.argwhere(tmp == edge)
            x, y = arg_list[0]
            # x = index // self.dim
            # y = index % self.dim
            # avoid deleting all connections of a input or output node
            # if there is only one connection left for input node, skip
            if y < m and np.count_nonzero(self.connect_mat[:, y]) == 1:
                continue
            # if there is only one connection left for output node, skip
            if x == self.dim - 1 and np.count_nonzero(self.connect_mat[x, :] == 1):
                continue
            self.connect_mat[x, y] = 0
            # self.connect_mat[y,x] = 0
            self.weight_mat[x, y] = 0
            # self.weight_mat[y,x] = 0
            count += 1
        return count

    def delete_nodes(self, num):
        """

        :param num: max number of nodes to be deleted
        :return: number of nodes deleted
        """
        count = 0
        hid_n_size = self.node_num - m - 1  # number of hidden nodes
        if hid_n_size <= 1:
            return -1

        d_num = random.randint(1, num)  # number of nodes to be deleted ######################

        while count < d_num:
            if self.node_num - m - 1 == 1:
                break

            if count == num:
                break

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

            # delete node
            # then delete connections and update matrices
            # discard row i and column i
            # Delete the i th column/row,
            # then add one column/row of zeros before the last column/row of the original matrix

            i = random.randint(0, self.hidden_nodes.count(1) - 1)  # the index of the hidden node to be deleted
            m_i = i + m  # convert index

            self.hidden_nodes = self.hidden_nodes[1:] + [0]
            # -----------update connection matrix--------------
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

        # update max_conn and min_conn
        self.max_conn = (m + m + self.hidden_nodes.count(1) - 1) * self.hidden_nodes.count(
            1) / 2 + m + self.hidden_nodes.count(1)
        self.min_conn = m - 1 + self.hidden_nodes.count(1)

        return count

    def cell_div(self, alpha):
        # add node only when there is vacant position
        if self.hidden_nodes.count(0) == 0:
            return -1
        # randomly choose a node to duplicate
        while True:
            parent_index = m + random.randint(0, len(self.hidden_nodes) - 1)
            if self.hidden_nodes[parent_index - m] == 1:
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

        # self.connect_mat[new_index, :] = self.connect_mat[parent_index, :]
        # self.connect_mat[:, new_index] = self.connect_mat[:, parent_index]
        self.connect_mat[new_index, new_index] = 0

        # 从前面连过来的weight不变，往后连的weight都要变
        tmp_mat = self.weight_mat
        tmp_mat = np.delete(tmp_mat, -2, 0)
        tmp_mat = np.insert(tmp_mat, new_index, self.weight_mat[parent_index, :], 0)
        tmp_mat = np.delete(tmp_mat, -2, 1)
        self.weight_mat = np.insert(tmp_mat, new_index, self.weight_mat[:, parent_index], 1)

        # edges from the beginning
        self.weight_mat[new_index, :parent_index + 1] = -alpha * self.weight_mat[parent_index, :parent_index + 1]
        # edges connecting forward
        self.weight_mat[parent_index + 1:, new_index] = -alpha * self.weight_mat[parent_index + 1:, parent_index]
        self.weight_mat[parent_index + 1:, parent_index] = (1 + alpha) * self.weight_mat[parent_index + 1:,
                                                                         parent_index]

        # lower triagularization
        self.connect_mat = np.tril(self.connect_mat, k=-1)
        self.weight_mat = np.tril(self.weight_mat, k=-1)

        self.max_conn = (m + m + self.hidden_nodes.count(1) - 1) * self.hidden_nodes.count(
            1) / 2 + m + self.hidden_nodes.count(1)
        self.min_conn = m - 1 + self.hidden_nodes.count(1)

        return 1


def sigmoid(z):
    # avoid runtime overflow
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = np.array(z, dtype=np.float128)
        return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    # Derivative of the sigmoid function.
    return sigmoid(z) * (1 - sigmoid(z))


def population_init(M, training_set, desired_out, epoch, weight_mat=None):
    # population = []
    # if number of individuals M larger than MAX_HID_NODES,
    # initialize MAX_HID_NODES number of networks with exactly 1,2,3,...,MAX_HID_NODES node(s)
    # if M > MAX_HID_NODES:
    #    population = [Network(MAX_HID_NODES, density, i+1, weight_mat=weight_mat) for i in range(MAX_HID_NODES)]
    #    population += [Network(MAX_HID_NODES, density, 1, weight_mat=weight_mat) for i in range(MAX_HID_NODES,M)]
    # else:
    population = [Network(MAX_HID_NODES, density, weight_mat=weight_mat) for i in range(M)]

    # is_success = [False for i in range(M)]
    init_error = [net.calc_error(training_set, desired_out) for net in population]
    after_error = [p.epoch_train(training_set, desired_out, 0.5, epoch) for p in population]
    percent = (np.array(init_error) - np.array(after_error)) / np.array(init_error)
    is_success = list(np.where(percent > 0.4, True, False))

    return population, is_success, after_error


def error_sort(population, is_success, error):
    # best to worst. For error: smallest to largest
    zipped = sorted(zip(population, is_success, error), key=lambda item: item[2])
    pop, suc, e = zip(*zipped)
    return list(pop), list(suc), list(e)


def choose(sorted_popu):
    M = len(sorted_popu)
    # initialize return values
    # the zeroth is the fittest
    p_idx = 0
    parent_c = sorted_popu[p_idx]
    rdm = random.random() * ((1 + M) * M / 2)
    l = list(range(M, 0, -1))
    for i in range(M, 0, -1):
        # probability for choosing (M-j)th individual:
        # P(M-j) = j/sum(1 to M)
        rdm -= i
        if rdm < 0:
            p_idx = M - i
            parent_c = sorted_popu[p_idx]
            break
    return parent_c, p_idx


def training_set_init(dim=5):
    """

    :param dim: dimension of the problem, by default 5.
    :return: x: X_train. y: Y_train
    """
    rows = 2 ** dim
    x = np.zeros([rows, dim])
    for col in range(x.shape[1]):
        x[:, col] = [i // (2 ** (dim - col - 1)) % 2 for i in range(rows)]
    y = (x.sum(1) + 1) % 2
    return x, y


def main(weight_mat=None):
    M = 20  # Population size
    EPOCH = 200
    learning_rate = 0.25  ###########################
    X_train, Y_train = training_set_init()
    ############################################
    epoch_count = 0

    # Generate population
    population, is_success, error = population_init(M, X_train, Y_train, epoch * 3)
    epoch_count += epoch * 3 * 20
    archive_1 = copy.deepcopy(population[0])
    archive_2 = copy.deepcopy(population[0])
    a1_flag = False  # record whether archive_1 is changed
    a2_flag = False
    # ------------------------------------------------------
    # Start training
#   print("----------------------> Start Training <-----------------------")
#    print("Population Size: {}\nLearning rate: {}".format(M, learning_rate))
    for t in range(EPOCH):
        # error = [net.calc_error(X_train, Y_train) for net in population]

        population, is_success, error = error_sort(population, is_success, error)
        parent, p_index = choose(population)
        # ------------------------------------------------------------------
        # STOPPING CRITERIA
        # ------------------------------------------------------------------
        if min(error) < 9:

            # All the networks that yield correct answers
            out = [indv for indv in population if
                   np.array_equal(Y_train, np.where(indv.get_answers(X_train) > 0.5, 1, 0))]
            if len(out) != 0:
                # Sort according to the number of hidden nodes.
                sorted_indv = sorted(out, key=lambda net: net.hidden_nodes.count(1))
                min_net = sorted_indv[0]

                if a1_flag is False or min_net.hidden_nodes.count(1) <= archive_1.hidden_nodes.count(1):
                    archive_1 = copy.deepcopy(min_net)
                    a1_flag = True

                # STOPPING CRITERIA
                if archive_1.hidden_nodes.count(1) <= 2:
                    return archive_1, epoch_count

            # All the networks with more than 20 test cases correctness
            out = [indv for indv in population if
                   (np.abs(Y_train - np.where(indv.get_answers(X_train) > 0.5, 1, 0))).sum() < 12]
            if len(out) != 0:
                # Sort according to the number of hidden nodes.
                sorted_indv = sorted(out, key=lambda i_net: i_net.hidden_nodes.count(1))
                min_net = sorted_indv[0]

                if a2_flag is False or min_net.hidden_nodes.count(1) <= archive_2.hidden_nodes.count(1):
                    archive_2 = copy.deepcopy(min_net)
                    a2_flag = True
            '''
            # print the number of wrong predictions on test cases
            print("number of wrong predictions")
            print([(np.abs(Y_train - np.where(i.get_answers(X_train) > 0.5, 1, 0))).sum() for i in population])
            print("number of hidden nodes")
            print([i.hidden_nodes.count(1) for i in population])  # the hidden nodes number
            '''
#        print("[EPOCH {}/{}]".format(t, EPOCH))
        '''
        print("-------------------[EPOCH {}/{}]-------------------------------------------------------------".format(t,
                                                                                                                     EPOCH))
        print("archive_1: {} {}, archive_2: {} {}".format(archive_1.hidden_nodes.count(1), a1_flag,
                                                          archive_2.hidden_nodes.count(1), a2_flag))
        print("All population error:\n{}".format(error))
        print(is_success)
        print("Current parent @ {}. Current error: {}".format(p_index, error[p_index]))
        print("Hidden nodes: {}".format(parent.hidden_nodes))
        print("Current Iteration: {}".format(epoch_count))
        '''
        offspring = copy.deepcopy(parent)
        error_before = error[p_index] = offspring.epoch_train(X_train, Y_train, learning_rate, epoch)
        population[p_index] = copy.deepcopy(offspring)
        epoch_count += epoch
        # ------------------------------------------------------------------
        # Typical training
        # ------------------------------------------------------------------
        if is_success[p_index]:
            # Train first
            error[p_index] = offspring.epoch_train(X_train, Y_train, learning_rate, epoch * 5)
            population[p_index] = copy.deepcopy(offspring)
            epoch_count += epoch * 5
#            print("error before-after: {}. error_after: {}".format(error_before - error[p_index], error[p_index]))
            if error_before - error[p_index] < 0.0001 * error_before:
                is_success[p_index] = False
#                print("set to False")
                continue

        # ------------------------------------------------------------------
        # Nodes deletion
        # ------------------------------------------------------------------
        else:  # delete nodes
#            print("Trying nodes deletion...")
            offspring = copy.deepcopy(parent)
            can_be_deleted = offspring.delete_nodes(2)  # n 不能比MAX_HID_NODE大
            error_after = offspring.epoch_train(X_train, Y_train, learning_rate, epoch * 3)
            epoch_count += epoch * 3
#            print("error before-after: {}. error_after: {}".format(error_before - error_after, error_after))

            # if better than the worst one, replace it
            if error_before - error_after > -0.2 and can_be_deleted >= 1:  # 0.001*error_before and can_be_deleted!=-1:
                error[-1] = error_after
                population[-1] = copy.deepcopy(offspring)
#                print("{} nodes deleted successfully, replace the worst individual".format(can_be_deleted))
                continue

            # ------------------------------------------------------------------
            # Connections deletion
            # ------------------------------------------------------------------
            else:  # delete connections
#                print("Trying connections deletion...")
                # offspring.calc_approx_impt() #######
                offspring = copy.deepcopy(parent)
                can_be_deleted = offspring.delete_conn(3)
                ###########
                error_after = offspring.epoch_train(X_train, Y_train, learning_rate, epoch * 2)
                epoch_count += epoch * 2
#                print("error before-after: {}. error_after: {}".format(error_before - error_after, error_after))

                # if better than the worst one, replace it
                if error_before - error_after > 0 and can_be_deleted >= 1:
                    error[-1] = error_after
                    population[-1] = copy.deepcopy(offspring)
#                    print("{} connections deleted, replace the worst individual".format(can_be_deleted))
                    continue

                # --------------------------------------------------------------
                # Connections addition & nodes addition
                # --------------------------------------------------------------
                else:  # add connections and nodes
#                    print("Trying connections addition & nodes addition...")
                    offspring = copy.deepcopy(parent)
                    offspring2 = copy.deepcopy(offspring)
                    success = offspring.add_connection(3)
                    success2 = offspring2.cell_div(-0.4)

                    if success > 0 and success2 > 0:
                        error_after_1 = offspring.epoch_train(X_train, Y_train, learning_rate, epoch)
                        error_after_2 = offspring2.epoch_train(X_train, Y_train, learning_rate, epoch)
                        epoch_count += epoch * 2
#                        print("both success. conn add error {}".format(error_after_1))
                        # replace the worst in one the population
                        if error_after_1 < error_after_2:
                            population[-1] = copy.deepcopy(offspring)
                            error[-1] = error_after_1
#                            print("{} connections added".format(success))
                        else:
                            population[-1] = copy.deepcopy(offspring2)
                            error[-1] = error_after_2
#                            print("{} nodes added".format(success2))

                    elif success <= 0 and success2 > 0:
                        error_after = offspring2.epoch_train(X_train, Y_train, learning_rate, epoch)
                        epoch_count += epoch
                        population[-1] = copy.deepcopy(offspring2)
                        error[-1] = error_after
#                        print("Nodes addition success")
#                        print("{} nodes added".format(success2))

                    elif success > 0 and success2 <= 0:
                        error_after = offspring.epoch_train(X_train, Y_train, learning_rate, epoch)
                        epoch_count += epoch
                        population[-1] = copy.deepcopy(offspring)
                        error[-1] = error_after
#                        print("Connections addition success, error {}".format(error_after))
#                        print("{} connection addition".format(success))

#                    print("error before-after: {}. error_after: {}".format(error_before - error_after, error_after))

    population, is_success, error = error_sort(population, is_success, error)
    candidate = copy.deepcopy(population[0])
    wrong = (np.abs(Y_train - np.where(candidate.get_answers(X_train) > 0.5, 1, 0))).sum()

#    print("candidate: {}".format(candidate.hidden_nodes.count(1)))
#    print("archive_1: {} flag: {}".format(archive_1.hidden_nodes.count(1), a1_flag))
#    print("archive_2: {} flag: {}".format(archive_2.hidden_nodes.count(1), a2_flag))

    result_1 = archive_1  # 全对的候选人
    result_2 = archive_2  # 不全对的候选人

    if a1_flag is True and wrong == 0 and candidate.hidden_nodes.count(1) < result_1.hidden_nodes.count(1):
        result_1 = candidate

    if a1_flag is False:
        if wrong == 0:
            result_1 = candidate
        else:
            result_1 = None

    if a2_flag is False:
        result_2 = None
    '''
    比较result1 和 result2
    都为None:
        return candidate
    result1 为None:
        return result2
    result2 为 None:
        return result1
    都不为None:
        比较result1 和 result2 的分数
        return 分数高的

    '''
    if result_1 is None and result_2 is None:
        return candidate, epoch_count
    elif result_1 is None:
        return result_2, epoch_count
    elif result_2 is None:
        return result_1, epoch_count
    else:
        out = result_1 if (2.0 / result_1.hidden_nodes.count(1) >= 1.0 / result_2.hidden_nodes.count(1)) else result_2
        return out, epoch_count


if __name__ == '__main__':

    start_time = time.time()
    if len(sys.argv) >= 3:
        if sys.argv[1] == '-s':
            random.seed(sys.argv[2])
            np.random.seed(random.randint(0, 10))

    output, epoch_count = main()

    x, y = training_set_init(5)
    wrong_count = (np.abs(y - np.where(output.get_answers(x) > 0.5, 1, 0))).sum()
#    print("Wrong predictions: {}".format(wrong_count))
#    print("Number of hidden nodes: {}".format(output.hidden_nodes.count(1)))

    output.epoch_train(x, y, 0.5, epoch * 10)
    epoch_count += epoch * 10
#    print("elapsed time {}".format(time.time() - start_time))

    indices = list(range(m, m + output.hidden_nodes.count(1))) + [-1]

#    print(output.weight_mat[indices, :-output.hidden_nodes.count(0) - 1])

    weight = output.weight_mat[indices, :-output.hidden_nodes.count(0) - 1]

    out_str = ""
    for row in weight:
        for i in row:
            out_str = out_str + str(i) + " "
        out_str = out_str.strip()
        out_str += "\n"
    out_str = out_str.strip("\n")
    print(out_str)
    print(epoch_count)
#    print(output.calc_error(x, y))
