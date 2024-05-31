import numpy as np
import time
import matplotlib.pyplot as plt

t = 1000 * time.time()

class Data_Owner:
    def __init__(self, m, d) -> None:
        self.m = m # number of tuples (Data Size)
        self.d = d # dimension of each tuple (Data Dimension)

        self.database = np.random.randint(0, 100, size=(self.m, self.d))

    def key_gen(self, c, epsilon, M_hat):
        self.c = c
        self.epsilon = epsilon
        self.eta = self.c + self.epsilon + self.d + 1

        # Fixed long term secrets
        s = np.random.randint(0, 100, size=(1, self.d + 1))
        tau = np.random.randint(0, 100, size=(1, self.c))

        # Per-tuple ephimeral secret
        self.v = np.random.randint(0, 100, size=(self.m, self.epsilon))

        self.secret_Key = (M_hat, s, tau)


    def data_encrypt(self):
        (M_hat, s, tau) = self.secret_Key
        encrypted_database = np.zeros((self.m, self.eta))
        for i in range(self.m):
            p_hat = np.concatenate((s, tau, self.v[i].reshape(1, self.epsilon)), axis=1) + np.concatenate((-2*self.database[i], np.array([np.linalg.norm(self.database[i])**2]), np.zeros(self.eta-self.d-1))).reshape(1, self.eta)
            p1 = np.matmul(p_hat, np.linalg.inv(M_hat))
            encrypted_database[i] = p1
        return encrypted_database
    
    # Ideally this shouldn't be implemented as it is a security breach but we do this on order to check correctness for the kNN algorithm
    def get_database(self):
        return self.database

class Query_User:
    def __init__(self, q, d) -> None:
        self.q = q # number of queries (Query Size)
        self.d = d # dimension of each tuple (Query Dimension)

        self.queries = np.random.randint(0, 100, size=(self.q, self.d))
    
    def key_gen(self, c, epsilon, M_hat):
        self.c = c
        self.epsilon = epsilon
        self.eta = self.c + self.epsilon + self.d + 1

        # Per-query ephemeral secrets
        self.r = np.random.randint(0, 100, size=(self.q, self.c))
        self.beta = np.random.randint(0, 100, size=(self.q, 1))

        self.secret_Key = M_hat
    
    def query_encrypt(self):
        M_hat = self.secret_Key
        q_hat = np.zeros((self.eta, self.q))
        for i in range(self.q):
            q_hat[:, i] = self.beta[i]*np.concatenate((self.queries[i], np.array([1]), self.r[i], np.zeros(self.epsilon))).reshape(1,self.eta)
        encrypted_query = np.matmul(M_hat, q_hat)
        return encrypted_query

    # Ideally this shouldn't be implemented as it is a security breach but we do this on order to check correctness for the kNN algorithm
    def get_queries(self):
        return self.queries


## k-NN Computation for the encrypted distances D(pi, q) and D(pj, q) would be as p1[i]q1 > p2[j]q1
def secure_kNN(encrypted_database, encrypted_query, k):
    encrypted_distances = np.matmul(encrypted_database, encrypted_query)
    
    #find indices of the k-smallest distances
    encrypted_indices = np.argsort(encrypted_distances, axis=0)

    return encrypted_indices[:k]

def kNN(database, query, k):
    distances = np.sum((database[:, np.newaxis, :] - query) ** 2, axis=2)
    distances = np.sqrt(distances)

    #find indices of the k-smallest distances
    distances = np.argsort(distances, axis=0)

    return distances[:k]


def main():
    d = 2   # dimension of each tuple (Data Dimension and Query Dimension)
    c, epsilon = 2, 5   # postive integers determining the dimensions of the secrets
    
    DO = Data_Owner(100, d)
    QU = Query_User(3, d)

    # M is an invertible matrix with (d+1+c+ϵ) rows/columns and with elements drawn uniformly at random from R.
    # The same matrix M must be available to both the Data Owner and the Query User in order to have a correct encryption and decryption
    M = np.random.randint(0, 100, size=(d+1+c+epsilon, d+1+c+epsilon)) 
    # Secret permutation function pi of length (d+1+c+ϵ)
    pi = np.random.permutation(d+1+c+epsilon)
    M_hat = M[:, pi]


    DO.key_gen(c, epsilon, M_hat)
    QU.key_gen(c, epsilon, M_hat)

    encrypted_database = DO.data_encrypt()
    encrypted_query = QU.query_encrypt()

    database = DO.get_database()
    queries = QU.get_queries()

    k = 10
    encrypted_kNN = secure_kNN(encrypted_database, encrypted_query, k)
    knn = kNN(database, queries, k)
    print(encrypted_kNN)
    print(knn)


    for i, query_point in enumerate(queries):
        r1 = np.linalg.norm(database[knn[k-1][i]]-query_point)

        x1 = r1*np.sin(np.linspace(0, 7, 20)) + query_point[0]
        y1 = r1*np.cos(np.linspace(0, 7, 20)) + query_point[1]
        
        plt.plot(x1, y1, 'k--')
    plt.plot(database[:, 0], database[:, 1], 'ro')
    plt.plot(queries[:, 0], queries[:, 1], 'bo')
    plt.plot(database[knn, 0], database[knn, 1], 'go')
    plt.savefig('aspe_data.png')
    plt.close()

    for i, query_point in enumerate(queries):
        r2 = np.linalg.norm(database[encrypted_kNN[k-1][i]]-query_point)

        x2 = r2*np.sin(np.linspace(0, 7, 20)) + query_point[0]
        y2 = r2*np.cos(np.linspace(0, 7, 20)) + query_point[1]
        
        plt.plot(x2, y2, 'k--')
    plt.plot(database[:, 0], database[:, 1], 'ro')
    plt.plot(queries[:, 0], queries[:, 1], 'bo')
    plt.plot(database[encrypted_kNN, 0], database[encrypted_kNN, 1], 'go')
    plt.savefig('aspe_encrypted_data.png')
    plt.close()

    
if __name__ == "__main__":
    main()

