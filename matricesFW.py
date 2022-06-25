# Set of functions for matrix completition using the Frank Wolfe algorithm and variants
import numpy as np
from scipy import sparse


# FW objective function
def FW_objective_function(diff_vec):
    return 0.5*(np.power(diff_vec,2).sum())

# Binary search for alpha stop
def alpha_binary_search(Zk, Dk, delta, max_value=1, min_value=0, tol=0.1):
    # Inizialization
    best_alpha = step = (max_value - min_value) / 2

    # Binary Search
    while step >= tol:

        testing_matrix = Zk + best_alpha * Dk
        testing_mat_nuclear_norm = LA.norm(testing_matrix, ord='nuc')
        step = step / 2

        if testing_mat_nuclear_norm > delta:
            best_alpha = best_alpha + step

        if testing_mat_nuclear_norm <= delta:
            best_alpha = best_alpha - step

    return best_alpha

def compute_best_beta(Zk, X_rated, Dk, idx_rows, idx_cols, alpha_stop, stepsize = 0.003, tol = 0.001):

  best_beta = alpha_stop / 2

  step = tol

  while step >= tol:

    Zk = Zk + best_beta * Dk                              # compute new Zk
    Z_rated = Zk[idx_rows, idx_cols]                      # take only known values
    diff_vec = np.array(Z_rated - X_rated)[0].sum()       # compute the difference vector
    #loss = 0.5*(np.power(diff_vec,2).sum())              # compute the loss # didn't take this one because it's always positive!

    step = diff_vec * stepsize
    best_beta = best_beta + step

    if best_beta <= 0:
      best_beta = 0
      break

    if best_beta >= alpha_stop:
      best_beta = alpha_stop
      break

  return best_beta



# Regular FW algorithm
def FrankWolfe(X, objective_function, delta, empties=0, printing = True, Z_init=None, max_iter=150, patience=1e-5):
    '''
    :param X: sparse matrix with ratings and 'empty values', rows - users, columns - books.
    :param objective_function: objective function that we would like to minimize with FW
    :param delta: Radius of the feasible's set ball
    :param empties (optional): Empty values of X are zeros ('zeros') or NaN ('nan'). Default = 'zeros'.
    :param Z_init (optional): In case we want to initialize Z with a known matrix- If not given, Z_init will be a zeros matrix. Default = None.
    :param max_iter (optional): max number of iterations for the method. Default = 150.
    :param patience (optional): once reached this tolerance provide the result. Default = 1e-5.
    :return: Zk: matrix of predicted ratings - it should be like X but with no 'empty values'
    :return: loss: objective function value with Zk
    '''

    # Get X indexes for not empty values
    if empties == 0:
        idx_ratings = np.argwhere(X != 0)
    elif empties == 'nan':
        idx_ratings = np.argwhere(~np.isnan(X))
    else:
        return print('Error: Empties argument', empties, 'not valid.')

    idx_rows = idx_ratings[:, 0]
    idx_cols = idx_ratings[:, 1]

    # Initialize Zk
    if Z_init == 'random uniform':
        Zk = np.random.uniform(low=0.01, high=1, size=X.shape)
    elif Z_init is not None:
        Zk = Z_init
    else:
        Zk = np.zeros(X.shape)

    # Create vectors with the not empty features of the sparse matrix
    X_rated = X[idx_rows, idx_cols]
    Z_rated = Zk[idx_rows, idx_cols]
    diff_vec = np.array(Z_rated - X_rated)[0]

    # Create needed variables
    diff_loss = patience + 1
    loss = objective_function(diff_vec)
    res_list = [loss]
    ranks = [np.linalg.matrix_rank(Zk)]
    it = 0
    while (diff_loss > patience) and (it < max_iter):

        # Gradient
        grad = sparse.csr_matrix((diff_vec, (idx_rows, idx_cols)))

        # SVD - Compute k = 1 singular values and its vectors, starting from the largest (which = 'LM')
        u_max, s_max, v_max = sparse.linalg.svds(grad, k=1, which='LM')  #

        # Update
        Zk_tilde = -delta * np.outer(u_max, v_max)

        alpha_k = 2 / (it + 2)  # alpha - as studied in class
        Zk = (1 - alpha_k) * Zk + alpha_k * Zk_tilde

        # Loss
        diff_vec = np.array(Zk[idx_rows, idx_cols] - X_rated)[0]
        new_loss = objective_function(diff_vec)

        # Improvement at this iteration
        diff_loss = np.abs(loss - new_loss)
        loss = new_loss

        rankZ = np.linalg.matrix_rank(Zk)
        if printing == True:
            if it == 1 or it % 50 == 0:
                print('Iteration:', it, 'Loss:', loss, 'Loss diff:', diff_loss, 'Rank(Zk): ', )

        # Count iteration
        it += 1

        res_list.append(loss)
        ranks.append(rankZ)

    return Zk, loss, res_list, ranks


# FW extended in face
def FW_inface(X, objective_function, delta, L = 1, D = None, gamma1 = 0, gamma2 = 1, empties = 0, max_iter=150, patience=1e-3, printing = True, delta_change = False):
    '''
    :param X: sparse matrix with ratings and 'empty values', rows - users, columns - books.
    :param objective_function: objective function that we would like to minimize with FW.
    :param delta: Radius of the feasible's set ball
    :param L: must be greater than 1
    :param D: if not inputed = 2*delta
    :param gamma1: constraint on B step
    :param gamma2: constraint on A step
    :param empties (optional): Empty values of X are zeros (0) or NaN ('nan'). Default = 0
    :param max_iter: max number of iterations for the method.
    :param patience: once reached this tolerance provide the result.
    :return: Z: matrix of predicted ratings - it should be like X but with no 'empty values'
            loss: difference between original values (X) and predicted ones (Z).
    '''

    # Get X indexes for not empty values
    if empties == 0:
        idx_ratings = np.argwhere(X != 0)
    elif empties == 'nan':
        idx_ratings = np.argwhere(~np.isnan(X))
    else:
        return print('Empties argument', empties, 'not valid.')

    idx_rows = idx_ratings[:,0]
    idx_cols = idx_ratings[:,1]

    # Initialize Z_{-1}
    Z_1 = np.zeros(X.shape)

    # Create vectors with the not empty features of the sparse matrix
    X_rated = X[idx_rows, idx_cols]
    Z_rated = Z_1[idx_rows, idx_cols]
    diff_vec_1 = np.array(Z_rated - X_rated)[0]

    # Initial gradient and Z0
    grad_1 = sparse.csr_matrix((diff_vec_1, (idx_rows, idx_cols)))
    u_max, s_max, v_max = sparse.linalg.svds(grad_1, k = 1, which='LM')
    Zk = -delta*np.outer(u_max,v_max)
    Z_rated = Zk[idx_rows, idx_cols]

    print('Initial Zk Rank: ', np.linalg.matrix_rank(Zk))

    # Initialize lower bound on the optimal objective function (f*)
    diff_vec = np.array(Z_rated - X_rated)[0]
    new_low_bound = 0

    # Set D
    if D is not None:
        D = D
    else:
        D = 2*delta

    rank_Z = np.linalg.matrix_rank(Zk)   # rank of Zk to find thin SVD size

    # Additional needed parameters
    diff_loss = patience + 1
    loss = objective_function(diff_vec)
    it = 0
    res_list = [objective_function(diff_vec_1), loss]
    ranks = [rank_Z]
    not_in_ball=False

    B_used = 0
    A_used = 0
    not_entered = 0
    regularFW = 0

    in_border = 0
    inside = 0
    outside = 0

    while (diff_loss > patience) and (it < max_iter):

        # Thin SVD
        U_thin, s_thin, Vh_thin = sparse.linalg.svds(Zk, k = rank_Z, which='LM')   # Compute k = rank singular values

        # Lower bound update
        low_bound = new_low_bound

        # Gradient
        grad = sparse.csr_matrix((diff_vec, (idx_rows, idx_cols)))

        # In-face direction with the away step strategy: two calculations depending of where Z lies within the feasible set
        if np.isclose(s_thin.sum(), delta, rtol=1e-3)==True: # Z in border (sum of singular values == radius of feasible set)
            in_border += 1
            G = 0.5 * (Vh_thin.dot(grad.T.dot(U_thin)) + U_thin.T.dot(grad.dot(Vh_thin.T)))
            # Obtain unitary eigenvector corresponding to smallest eigenvalue of G
            eigvalues, eigvectors = np.linalg.eig(G)  #find the eigenvalues
            min_eig = np.argmin(eigvalues)  #find the index of the smallest eigenvalue
            u = eigvectors[:, min_eig]  #take the eigenvector corresponding to the smallest eigenvalue

            # Update value
            M = np.outer(u,u.T)
            Zk_tilde = delta*U_thin.dot(M.dot(Vh_thin))
            Dk = Zk - Zk_tilde

            inv_s_thin = np.diag(1/s_thin)
            alpha_stop = 1/(delta*u.T.dot(inv_s_thin.dot(u))-1)

        elif s_thin.sum() < delta: #inside
            inside += 1
            u_max, s_max, v_max = sparse.linalg.svds(grad, k = 1, which='LM')
            Zk_tilde = delta*np.outer(u_max,v_max)
            Dk = Zk - Zk_tilde
            #BINARY SEARCH
            alpha_stop = alpha_binary_search(Zk, Dk, delta)

        else:
            not_in_ball = True
            outside += 1

        #nuclear_norm = s_thin.sum()
        # We check if Zk is inside the feasible set and the rank in order to decide which type steps to do
        if rank_Z > 1 and not_in_ball==False:
            if delta_change is True:
                delta = delta * (1 -0.03)
            Z_B = Zk + alpha_stop*Dk
            diff_vec_B = np.array(Z_B[idx_rows, idx_cols] - X_rated)[0]
            if 1/(objective_function(diff_vec_B)-low_bound) >= (1/(loss-low_bound)+gamma1/(2*L*D**2)):
              # 1. Move to a lower dimensional face
                B_used += 1
                Zk = Z_B

            else:

                beta = compute_best_beta(Zk, X_rated, Dk, idx_rows, idx_cols, alpha_stop)
                Z_A = Zk + beta*Dk
                diff_vec_A = np.array(Z_A[idx_rows, idx_cols] - X_rated)[0]

                if 1/(objective_function(diff_vec_A)-low_bound) >= (1/(loss-low_bound)+gamma2/(2*L*D**2)):
                    # 2. Stay in the current face
                    A_used += 1
                    Zk = Z_A
                else:
                    # 3. Do a regular FW step and update the lower bound
                    regularFW += 1
                    #Zk update
                    u_max, s_max, v_max = sparse.linalg.svds(grad, k = 1, which='LM')
                    Zk_tilde = -delta*np.outer(u_max,v_max)

                    direction_vec = Zk_tilde.flatten() - Zk.flatten()
                    grad = grad.toarray()
                    wolfe_gap = grad.T.flatten() * direction_vec
                    B_w = loss + wolfe_gap.sum()

                    if low_bound >= B_w:
                        new_low_bound = low_bound
                    else:
                        new_low_bound = B_w

                    # Z_(k+1)
                    alpha_k = 2/(it+2)
                    Zk = (1-alpha_k)*Zk + alpha_k*Zk_tilde

        else:

            # 3. Do a regular FW step and update the lower bound
            not_entered += 1
            #Zk update
            grad = sparse.csr_matrix((diff_vec, (idx_rows, idx_cols)))
            u_max, s_max, v_max = sparse.linalg.svds(grad, k = 1, which='LM')
            Zk_tilde = -delta*np.outer(u_max,v_max)

            direction_vec = Zk_tilde.flatten() - Zk.flatten()
            grad = grad.toarray()
            wolfe_gap = grad.T.flatten() * direction_vec
            B_w = loss + wolfe_gap.sum()

            if low_bound >= B_w:
                new_low_bound = low_bound
            else:
                new_low_bound = B_w

            # Z_(k+1)
            alpha_k = 2/(it+2)
            Zk = (1-alpha_k)*Zk + alpha_k*Zk_tilde

            not_in_ball=False

        # Loss
        diff_vec = np.array(Zk[idx_rows, idx_cols] - X_rated)[0]
        new_loss = objective_function(diff_vec)

        # Improvement at this iteration
        diff_loss = np.abs(loss - new_loss)
        loss = new_loss

        rank_Z = np.linalg.matrix_rank(Zk)   # rank of Zk to find thin SVD size

        # Count iteration
        it += 1

        res_list.append(loss)
        ranks.append(rank_Z)
        if printing == True:
          if it % 10 == 0 or it == 1:
            print('Iteration:', it, 'Loss:', loss, 'Loss diff:', diff_loss, 'Rank(Z): ', rank_Z)

    print('Iteration:', it, 'Loss:', loss, 'Loss diff:', diff_loss, 'Rank(Z): ', rank_Z)
    print('Went to lower dim face:', B_used)
    print('Stayed:', A_used)
    print('Regular:', regularFW)
    print('Not entered if:', not_entered)

    print('in_border:', in_border)
    print('Inside ball:', inside)
    print('Outside ball:', outside)
    return Zk, loss, res_list, ranks
