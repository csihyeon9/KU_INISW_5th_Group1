import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter

def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))

    return matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

import numpy as np
import scipy.sparse as sp

class Data:
    def __init__(self, data, shuffle=False, n_node=None):
        # Ensure data is properly processed
        if isinstance(data, list) and len(data) > 0:
            self.raw = np.asarray(data[0], dtype=object)
            self.targets = np.asarray(data[1])
        else:
            raise ValueError("Invalid data format. Expected a list with at least two elements.")

        # Handle node count
        if n_node is None:
            n_node = self._get_max_node()

        # Safely create adjacency matrix
        self.adjacency = self._create_adjacency_matrix(n_node)
        
        self.n_node = n_node
        self.length = len(self.raw)
        self.shuffle = shuffle

        # Optional shuffling
        if shuffle:
            self._shuffle_data()

    def _get_max_node(self):
        """Safely find the maximum node in the dataset"""
        max_node = 0
        for session in self.raw:
            if isinstance(session, list):
                max_node = max(max_node, max(session) if session else 0)
        return max_node

    def _create_adjacency_matrix(self, n_node):
        """Create adjacency matrix with comprehensive error handling and diagnostics"""
        try:
            # First, print out some diagnostic information
            print("Raw data shape:", self.raw.shape)
            print("Number of nodes:", n_node)

            # Implement data_masks with more explicit error handling
            def data_masks(all_sessions, n_node):
                # Create a sparse matrix to represent session-node interactions
                masks = sp.lil_matrix((len(all_sessions), n_node))
                
                for i, session in enumerate(all_sessions):
                    # Ensure session is converted to a list if it's not
                    if not isinstance(session, list):
                        session = list(session)
                    
                    for item in session:
                        # Validate item and ensure it's within node range
                        if isinstance(item, (int, np.integer)) and 0 < item <= n_node:
                            masks[i, item-1] = 1
                
                return masks.tocsr()

            # Create the masks matrix
            H_T = data_masks(self.raw, n_node)
            
            # Print shapes for diagnostics
            print("H_T shape:", H_T.shape)

            # Safe matrix computations with extensive error checking
            epsilon = 1e-10
            
            # Compute row sums safely
            row_sums = H_T.sum(axis=1)
            
            # Ensure row_sums is a 1D array
            row_sums = np.asarray(row_sums).flatten()
            
            # Create normalized matrix with safe division
            row_sums_inv = np.zeros_like(row_sums, dtype=float)
            non_zero_mask = row_sums > 0
            row_sums_inv[non_zero_mask] = 1.0 / (row_sums[non_zero_mask] + epsilon)
            
            # Transpose and multiply
            BH_T = H_T.T.multiply(row_sums_inv.reshape(1, -1))
            
            # Similar process for H
            H = H_T.T
            col_sums = H.sum(axis=1)
            col_sums = np.asarray(col_sums).flatten()
            col_sums_inv = np.zeros_like(col_sums, dtype=float)
            non_zero_mask = col_sums > 0
            col_sums_inv[non_zero_mask] = 1.0 / (col_sums[non_zero_mask] + epsilon)
            
            DH = H.T.multiply(col_sums_inv.reshape(1, -1))
            DH = DH.T

            # Ensure matrix multiplication is possible
            print("BH_T shape:", BH_T.shape)
            print("DH shape:", DH.shape)

            # Compute final adjacency matrix
            DHBH_T = sp.csr_matrix(DH @ BH_T)
            
            return DHBH_T

        except Exception as e:
            print(f"Detailed error in creating adjacency matrix: {e}")
            # Return an identity matrix as a fallback
            return sp.eye(n_node, dtype=float)

    def _data_masks(self, all_sessions, n_node):
        """Create data masks for sessions"""
        # Implement data_masks function logic here
        # This is a placeholder and might need to be adapted from your original implementation
        masks = sp.lil_matrix((len(all_sessions), n_node))
        for i, session in enumerate(all_sessions):
            for item in session:
                if 0 < item <= n_node:
                    masks[i, item-1] = 1
        return masks.tocsr()

    def _shuffle_data(self):
        """Shuffle data consistently"""
        shuffled_arg = np.arange(self.length)
        np.random.shuffle(shuffled_arg)
        self.raw = self.raw[shuffled_arg]
        self.targets = self.targets[shuffled_arg]

    def get_overlap(self, sessions):
        """Compute session overlap with safe calculations"""
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i]) - {0}
            
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j]) - {0}
                
                # Safe union and intersection
                ab_set = seq_a.union(seq_b)
                overlap = seq_a.intersection(seq_b)
                
                # Compute overlap ratio safely
                matrix[i][j] = float(len(overlap)) / float(len(ab_set)) if ab_set else 0.0
                matrix[j][i] = matrix[i][j]
        
        # Add diagonal of 1s
        matrix += np.diag([1.0] * len(sessions))
        
        # Safe degree computation
        degree = np.sum(matrix, axis=1)
        degree_inv = np.zeros_like(degree)
        non_zero_mask = degree != 0
        degree_inv[non_zero_mask] = 1.0 / degree[non_zero_mask]
        degree = np.diag(degree_inv)
        
        return matrix, degree

    def generate_batch(self, batch_size):
        """Generate batches with safe splitting"""
        if self.shuffle:
            self._shuffle_data()
        
        n_batch = int(np.ceil(self.length / batch_size))
        slices = []
        
        for i in range(n_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.length)
            slices.append(np.arange(start, end))
        
        return slices

    def get_slice(self, index):
        """Get data slice with padding"""
        items, num_node = [], []
        inp = self.raw[index]
        
        # Compute max node count for padding
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        
        session_len, reversed_sess_item, mask = [], [], []
        
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            
            # Pad sessions to max length
            padded_session = session + [0] * (max_n_node - len(nonzero_elems))
            items.append(padded_session)
            
            # Create mask for valid elements
            session_mask = [1] * len(nonzero_elems) + [0] * (max_n_node - len(nonzero_elems))
            mask.append(session_mask)
            
            # Reverse session with padding
            reversed_sess_item.append(list(reversed(session)) + [0] * (max_n_node - len(nonzero_elems)))
        
        return np.array(self.targets)[index] - 1, session_len, items, reversed_sess_item, mask
