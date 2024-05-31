import numpy as np
import time
import matplotlib.pyplot as plt

t = 1000 * time.time()

class Data_Owner:
    def __init__(self, n, d) -> None:
        self.n = n # number of tuples (Data Size)
        self.d = d # dimension of each tuple (Data Dimension)

        self.database = np.random.randint(0, 100, size=(self.n, self.d))

    def key_gen(self, c, epsilon, M_base):
        self.c = c
        self.epsilon = epsilon
        self.eta = self.c + self.epsilon + self.d + 1

        # Fixed long term secrets
        s = np.random.randint(0, 100, size=(1, self.d + 1))
        w = np.random.randint(0, 100, size=(1, self.c))

        # Per-tuple ephimeral secret
        self.z = np.random.randint(0, 100, size=(1, self.epsilon))
        # Per-query ephemeral secret
        self.beta2 = np.random.randint(0, 100, size=(1, 1))

        self.secret_Key = (M_base, s, w)


    def data_encrypt(self):
        (M_base, s, w) = self.secret_Key
        encrypted_database = {'data':np.zeros((self.n, self.eta)), 'Max_norm':0}
        
        max_norm = 0
        for i in range(self.n):
            p_hat = np.concatenate((s, w, self.z), axis=1) + np.concatenate((-2*self.database[i], np.array([np.linalg.norm(self.database[i])**2]), np.zeros(self.c + self.epsilon))).reshape(1, self.eta)
            p1 = np.matmul(p_hat, np.linalg.inv(M_base))        # Time = O(eta^2)
            encrypted_database['data'][i] = p1
            if np.linalg.norm(self.database[i]) > max_norm:
                max_norm = np.linalg.norm(self.database[i])
        # Time = O(n * eta^2)
        encrypted_database['Max_norm'] = max_norm
        
        self.encrypted_database = encrypted_database
        return encrypted_database
    
    def query_encrypt(self, query, uid):
        (M_base, s, w) = self.secret_Key
        encrypted_query = np.zeros((self.eta, self.eta))
        
        temp_sec_matrix = {'user_id':0, 'matrix': np.zeros((self.eta, self.eta))}
        q_dot = query['queries'][0]
        q_max = max(q_dot)
        # print(q_max, self.encrypted_database['Max_norm'])
        q_max = 0.001

        # print(q_max, self.encrypted_database['Max_norm'])
        M_t = np.random.randint(100*q_max, q_max*1000, size=(self.eta, self.eta))    
        for a in range(self.eta):
            M_t[a][a] = np.random.randint(100*self.encrypted_database['Max_norm'], 1000*self.encrypted_database['Max_norm'])
        M_sec = np.matmul(M_t, M_base) 
        
        x = np.random.randint(0,100,size=(1, self.c))
        q1 = np.concatenate((q_dot.reshape(1, self.d), np.array([1]).reshape(1,1), x, np.zeros(self.epsilon).reshape(1, self.epsilon)), axis=1).reshape(1, self.eta)
        qnn = np.eye(self.eta)
        for a in range(self.eta):
            qnn[a][a] = q1[0, a]
        # print(q1) 
        # print(qnn)

        E = np.random.randint(100*q_max, q_max*1000,size=(self.eta, self.eta)) # Error matrix
        # print(E)
        # print(M_base)
        # print(M_t)
        # print(M_sec)
        q_hat = self.beta2*(np.matmul(M_sec, qnn) + E) # Time = O(eta^3)
        
        encrypted_query = q_hat
        temp_sec_matrix['user_id'] = uid
        temp_sec_matrix['matrix'] = M_t

        return temp_sec_matrix, encrypted_query

    # Ideally this shouldn't be implemented as it is a security breach but we do this on order to check correctness for the kNN algorithm
    def get_database(self):
        return self.database

class Query_User:
    def __init__(self, query, d) -> None:
        self.query = query # query point
        self.d = d # dimension of each tuple (Query Dimension)
    
    def key_gen(self, c, epsilon, M_base, N):
        self.c = c
        self.epsilon = epsilon
        self.eta = self.c + self.epsilon + self.d + 1

        # Per-query ephemeral secrets
        self.r = np.random.randint(0, 100, size=(1, self.c))
        self.beta1 = 1

        self.secret_Key = (M_base, N)

    
    def query_encrypt1(self, uid):
        (M_base, N) = self.secret_Key
        
        encrypted_query = {'user_id':0, 'queries':np.zeros((1, self.d))}
        q_dot = self.beta1*np.matmul(self.query.reshape(1, self.d), N)
        encrypted_query['user_id'] = uid
        encrypted_query['queries'] = q_dot

        return encrypted_query
    
    def query_encrypt2(self, queries):
        (M_base, N) = self.secret_Key
        
        N1 = np.eye(self.eta)
        for a in range(self.d):
            N1[a][a] = N[a][a]
        
        q_enc = np.matmul(queries, np.linalg.inv(N1)) # Time = O(eta^3)

        # print(q_enc)
        q_vec = np.sum(q_enc, axis=1)
        # print(q_vec)
        return q_vec

    # Ideally this shouldn't be implemented as it is a security breach but we do this on order to check correctness for the kNN algorithm
    def get_queries(self):
        return self.query
    

class Cloud_Service_Provider:
    # def __init__(self) -> None:
    #     pass

    def data_encrpyt(self, encrypted_database, temp_matrix):
        # Time = O(n * eta^2)
        temp_database = np.matmul(encrypted_database['data'], np.linalg.inv(temp_matrix['matrix']))
        return temp_database

    ## k-NN Computation for the encrypted distances D(pi, q) and D(pj, q) would be as p1[i]q1 > p2[j]q1 
    def secure_kNN(self, encrypted_database, encrypted_query, k):
        encrypted_distances = np.matmul(encrypted_database, encrypted_query)
        
        #find indices of the k-smallest distances
        encrypted_indices = np.argsort(encrypted_distances, axis=0) # Time = O(n*eta*log(k))
        encrypted_distances = np.sort(encrypted_distances, axis=0)

        return encrypted_indices[:k]

    def kNN(self, database, query, k):
        distances = np.sum((database[:, np.newaxis, :] - query) ** 2, axis=2)
        distances = np.sqrt(distances)

        #find indices of the k-smallest distances
        distances = np.argsort(distances, axis=0)

        return distances[:k]


def main():
    d = 2   # dimension of each tuple (Data Dimension and Query Dimension)
    c, epsilon = 1, 1   # postive integers determining the dimensions of the secrets
    
    query_point = np.random.randint(0, 100, size=(1, d))


    DO = Data_Owner(100, d)
    QU = Query_User(query_point, d)
    CSP = Cloud_Service_Provider()

    # M_base is an invertible matrix with (d+1+c+ϵ) rows/columns and with elements drawn uniformly at random from R.
    # The same matrix M_base must be available to both the Data Owner and the Query User in order to have a correct encryption and decryption    
    M_base = np.random.randint(0, 100, size=(d+1+c+epsilon, d+1+c+epsilon)) 
    N = np.diag(np.diag(np.random.randint(1, 100, size=(d, d))))
    

    DO.key_gen(c, epsilon, M_base)
    QU.key_gen(c, epsilon, M_base, N)

    encrypted_database = DO.data_encrypt()
    encrypted_query = QU.query_encrypt1(uid=1)

    temp_matrix, encrypted_query = DO.query_encrypt(encrypted_query, uid=1)
    encrypted_database = CSP.data_encrpyt(encrypted_database, temp_matrix)
    encrypted_query = QU.query_encrypt2(encrypted_query)


    database = DO.get_database()
    queries = QU.get_queries()

    k = 10
    encrypted_kNN = CSP.secure_kNN(encrypted_database, encrypted_query, k)
    knn = CSP.kNN(database, queries, k)
    # print(encrypted_kNN)
    # print(knn[:, 0])

    r1 = np.linalg.norm(database[knn[k-1]]-query_point)
    r2 = np.linalg.norm(database[encrypted_kNN[k-1]]-query_point)

    x1 = r1*np.sin(np.linspace(0, 7, 18)) + query_point[0,0]
    y1 = r1*np.cos(np.linspace(0, 7, 18)) + query_point[0,1]

    x2 = r2*np.sin(np.linspace(0, 7, 18)) + query_point[0,0]
    y2 = r2*np.cos(np.linspace(0, 7, 18)) + query_point[0,1]
    
    plt.plot(x1, y1, 'k--')
    plt.plot(database[:, 0], database[:, 1], 'ro')
    plt.plot(queries[:, 0], queries[:, 1], 'bo')
    plt.plot(database[knn[:, 0], 0], database[knn[:, 0], 1], 'go')
    plt.savefig('secureKNN_data.png')
    plt.close()


    plt.plot(x2, y2, 'k--')
    plt.plot(database[:, 0], database[:, 1], 'ro')
    plt.plot(queries[:, 0], queries[:, 1], 'bo')
    plt.plot(database[encrypted_kNN, 0], database[encrypted_kNN, 1], 'go')
    plt.savefig('secureKNN_encrypted_data.png')
    plt.close()

    
if __name__ == "__main__":
    main()

