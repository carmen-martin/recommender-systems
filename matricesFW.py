# Set of functions for matrix completition using the Frank Wolfe algorithm and variants

# IMPORTING THE LIBRARIES
import numpy as np            # for matrix computation
from scipy import sparse      # to generate - and manage - sparse matrix more efficiently
import scipy.sparse.linalg    # to execute advance computation on matrices like the norm and the gradient


# OBJECTIVE FUNCTION
def FW_objective_function(diff_vec):
    ''' COMPUTE delta f(x)
    
    DESCRIPTION:
    This function compute the loss of the current iteration using a vector with the differences between X_rated (the observed values) and Z_rated (the predicted observed values). 
    It is used in both the regular Frank-Wolfe and the InFace Frank Wolfe.
    
    INPUT:
    diff_vec -- np.array of dimension [p] -- An array with the differences between the observed values (X_rated) and the predicted observed values (Z_rated)
    
    OUTPUT:
    float -- The loss of the current iteration
    
    '''
    return 0.5*(np.power(diff_vec,2).sum()) # Take the vector with the differences, square its elements, sum them and halve the result

# BINARY SEARCH FOR ALPHA
def alpha_binary_search(Zk, Dk, delta, max_value=1, min_value=0, tol=0.1):
    ''' COMPUTE THE BEST ALPHA STOP
    
    DESCRIPTION:
    This function compute the best alpha stop as reported in the Freund-Grigas-Mazumder paper. 
    The starting value is set in the middle of the search space (in our case 0.5).
    It's possible, although not advised, to change the dimension of the search space by using max_value and min_value.
    We also input a tolerance parameter to avoid enter an infinite loop (never set this parameter to 0).
        Increasing the tolerance means to speed up the algorithm and to raise the loss.
    
    INPUT:
    Zk -- np. matrix of dimension [m, n] -- The current x
    Dk -- np.matrix of dimension [m, n] -- The direction vector
    Delta -- integer -- The radius of the ball
    max_value -- float -- The maximum value of the search space
    min_value -- float -- The minimum value of the search space
    tol -- float -- The tolerance of the algorihtm
    
    
    OUTPUT:
    best_alpha -- float -- The best alpha found
    
    '''
    
    # Initialization
    best_alpha = step = (max_value - min_value) / 2   # place the point in the middle of the search space

    # Binary Search
    while step >= tol:       # verify that the step is meaningful

        testing_matrix = Zk + best_alpha * Dk                            # compute the matrix
        testing_mat_nuclear_norm = LA.norm(testing_matrix, ord='nuc')    # compute the nuclear norm of the matrix
        step = step / 2      # halve the step

        # choose the direction of the next step based on the nuclear norm and the delta
        if testing_mat_nuclear_norm > delta:
            best_alpha = best_alpha + step

        if testing_mat_nuclear_norm <= delta:
            best_alpha = best_alpha - step

    return best_alpha

# EXACT LINE SEARCH FOR BETA
def compute_best_beta(Zk, X_rated, Dk, idx_rows, idx_cols, alpha_stop, stepsize = 0.003, tol = 0.001):
    ''' COMPUTE THE BEST BETA
    
    DESCRIPTION:
    The function find the best beta based on alpha and the difference vector between the actual observed values (X_rated) and the predicted observed values (Z_rated).
        We didn't use the objective function because, by construction, it's always positive and that would mean moving only in one direction.
    To increase/decrease the speed and accuracy of the algorithm it's possible to set different values for the stepsize and the tolerance.
    
    INPUT:
    Zk -- np.matrix of dimension [m, n] -- The current x
    X_rated -- np.array of dimension [p] -- The actual observed values
    Dk -- np.matrix of dimension [m, n] -- The directions of the next iterate
    idx_rows -- np.array of dimension [p] -- The row indices of the observed values. Used to find Z_rated.
    idx_cols -- np.array of dimension [p] -- The column indices of the observed values. Used to find Z_rated.
    alpha_stop -- float -- The alpha found in alpha_binary_search
    stepsize -- float -- The size of each step taken by the algorithm. Increasing means losing accuracy and gaining speed.
    tol -- float -- The tolerance of the algorithm. Increasing means losing accuracy and gaining speed.
    
    OUTPUT:
    best_beta - scalar - The best beta found
    
    '''
    #Initialization
    
    best_beta = alpha_stop / 2 # start in the middle of the search space
    step = tol                 # set up the step so we can enter the cycle at least one time

    while step >= tol:         # check that we are getting meaningful steps

        Zk = Zk + best_beta * Dk                              # compute new Zk
        Z_rated = Zk[idx_rows, idx_cols]                      # take only known values
        diff_vec = np.array(Z_rated - X_rated)[0].sum()       # compute the difference vector

        step = diff_vec * stepsize          # compute the step
        best_beta = best_beta + step        # compute the best beta

        # set default values if the best_beta is outside of the search space
        if best_beta <= 0:
            return 0

        if best_beta >= alpha_stop:
            return alpha_stop

    return best_beta

# Regular FW algorithm
def FrankWolfe(X, objective_function, delta, empties=0, printing = True, Z_init=None, max_iter=150, patience=1e-5):
    ''' PREDICTING VALUES USING THE REGULAR FRANK-WOLFE ALGORITHM
    
    DESCRIPTION:
    The algorithm fill the matrix Zk based on some observed values (given by the sparse matrix X) while minimizing an objective function.
    By changing the delta - the radius of the ball - it possible increase/decrease the accuracy.
    
    INPUT:
    X -- np.matrix of dimension [m, n] -- sparse matrix with ratings and 'empty values', rows - users, columns - books.
    objective_function -- function -- objective function that we would like to minimize with FW
    delta -- float -- Radius of the feasible's set ball
    [optional] empties -- string -- Empty values of X are zeros ('zeros') or NaN ('nan'). Default = 'zeros'.
    [optional] Z_init -- np.matrix of dimension [m, n] In case we want to initialize Z with a known matrix- If not given, Z_init will be a zeros matrix. Default = None.
    [optional] max_iter -- integer -- Max number of iterations for the method. Default = 150.
    [optional] patience -- float -- Once reached this tolerance provide the result. Default = 1e-5.
    
    OUTPUT:
    Zk -- np.matrix of dimension [m,n] -- Matrix of predicted ratings: it should be like X but with no 'empty values'
    loss -- float -- The error associated with the final Zk
    res_list -- list of lenght [iterations] -- A list containing the losses at each iteration
    ranks -- list of lenght [iterations] -- A list containing the rank of each predicted Zk
    
    '''

    # Get X indexes for not empty values
    if empties == 0:
        idx_ratings = np.argwhere(X != 0)
    elif empties == 'nan':
        idx_ratings = np.argwhere(~np.isnan(X))
    else:
        return print('Error: Empties argument', empties, 'not valid.')

    # Separate the indexes coordinates
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
    
    # Compute the vector containing the differences between initial Zk matrix and the observed matrix X
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
        alpha_k = 2 / (it + 2)  # alpha
        Zk = (1 - alpha_k) * Zk + alpha_k * Zk_tilde

        # Loss
        diff_vec = np.array(Zk[idx_rows, idx_cols] - X_rated)[0]  # compute the vector of differences between the predicted and the actual observed entries
        new_loss = objective_function(diff_vec) # compute the loss of Zk

        # Improvement at this iteration
        diff_loss = np.abs(loss - new_loss)
        loss = new_loss

        rankZ = np.linalg.matrix_rank(Zk)
        if printing == True:
            if it == 1 or it % 50 == 0:
                print('Iteration:', it, 'Loss:', loss, 'Loss diff:', diff_loss, 'Rank(Zk): ', )

        # Count iteration
        it += 1

        # Appending results
        res_list.append(loss)
        ranks.append(rankZ)

    return Zk, loss, res_list, ranks


# FW extended in face
def FW_inface(X, objective_function, delta, L = 1, D = None, gamma1 = 0, gamma2 = 1, empties = 0, max_iter=150, patience=1e-3, printing = True, delta_change = False):
    ''' PREDICT VALUES USING IN-FACE FRANK-WOLFE
    
    DESCRIPTION:
    The algorithm fill a sparse matrix by using the In-Face Frank-Wolfe method.
    It's possible to control the algorithm performance by using several parameters.
    Increasing gamma2 will lead to less iterations where the algorithm stay in the same face.
    Increasing gamma1 will lead to less iterations where the algorithm go to a lower-dimensional face.
    Changing delta influence the number of times the algorihtm will be inside or outside the ball.
    Using the delta increaser will lead to more accurate, higher ranked matrices.
    
    INPUT:
    X -- sparse matrix of dimension [m, n] -- A matrix with some observed values
    objective_function -- function -- objective function that we would like to minimize with the In-Face FW
    delta -- float -- Radius of the feasible's set ball
    [optional] L -- float -- Lipshitz constant of the gradient of f(x). Default = 1
    [optional] D -- float -- Diameter of the feasible set. Default = 2 * delta
    [optional] gamma1 -- float -- Constraint on the B step. Must be 0 <= gamma1 <= gamma2. Default = 0
    [optional] gamma2 -- float -- constraint on the A step. Must be 0 <= gamma2 <= 1. Default = 1
    [optional] empties -- 0 or 'nan' -- Empty values of X are zeros (0) or NaN ('nan'). Default = 0
    [optional] max_iter - integer -- Maximum number of iterations. Default = 150
    [optional] patience -- float -- Once reached this tolerance provide the result. Default = 1e-3
    [optional] printing -- bool -- Option to print the results while executing the algorithm. Default = True
    [optional] delta_change -- bool -- Option to increase the delta at each iteration. Default = False
    
    OUTPUT:
    Zk -- np.matrix of dimension [m,n] -- Matrix of predicted ratings: it should be like X but with no 'empty values'
    loss -- float -- The error associated with the final Zk
    res_list -- list of lenght [iterations] -- A list containing the losses at each iteration
    ranks -- list of lenght [iterations] -- A list containing the rank of each predicted Zk
    
    '''

    # Get X indexes for not empty values
    if empties == 0:
        idx_ratings = np.argwhere(X != 0)
    elif empties == 'nan':
        idx_ratings = np.argwhere(~np.isnan(X))
    else:
        return print('Empties argument', empties, 'not valid.')
    
    # Separate the indexes coordinates
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
    
    # Add counters
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

        elif s_thin.sum() < delta: # Z inside the ball
            
            inside += 1
            u_max, s_max, v_max = sparse.linalg.svds(grad, k = 1, which='LM')
            Zk_tilde = delta*np.outer(u_max,v_max)
            Dk = Zk - Zk_tilde
            alpha_stop = alpha_binary_search(Zk, Dk, delta)

        else:
            
            not_in_ball = True
            outside += 1

        # We check if Zk is inside the feasible set and the rank in order to decide which type steps to do
        if rank_Z > 1 and not_in_ball==False:
            
            # Change the delta at each iteration
            if delta_change is True:
                delta = delta * (1 + 0.03)
            
            # Compute Z_B and the differences with the observed values
            Z_B = Zk + alpha_stop*Dk
            diff_vec_B = np.array(Z_B[idx_rows, idx_cols] - X_rated)[0]
            
            if 1/(objective_function(diff_vec_B)-low_bound) >= (1/(loss-low_bound)+gamma1/(2*L*D**2)):
                
                # 1. Move to a lower dimensional face
                B_used += 1
                Zk = Z_B

            else:

                # Compute Beta
                beta = compute_best_beta(Zk, X_rated, Dk, idx_rows, idx_cols, alpha_stop)
                
                # Use Beta to compute Z_A and the differences with the observed values
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
                    
                    # Compute SVD
                    u_max, s_max, v_max = sparse.linalg.svds(grad, k = 1, which='LM')
                    
                    # Compute Dk
                    Zk_tilde = -delta*np.outer(u_max,v_max)
                    direction_vec = Zk_tilde.flatten() - Zk.flatten()

                    # Compute new boundary
                    grad = grad.toarray()
                    wolfe_gap = grad.T.flatten() * direction_vec
                    B_w = loss + wolfe_gap.sum()

                    # Take the maximum bound
                    if low_bound >= B_w:
                        new_low_bound = low_bound
                    else:
                        new_low_bound = B_w

                    # Copmute Z_(k+1)
                    alpha_k = 2/(it+2)
                    Zk = (1-alpha_k)*Zk + alpha_k*Zk_tilde

        else:

            # 3. Do a regular FW step and update the lower bound
            not_entered += 1
            
            # Compute the gradient
            grad = sparse.csr_matrix((diff_vec, (idx_rows, idx_cols)))
            
            # Compute SVD
            u_max, s_max, v_max = sparse.linalg.svds(grad, k = 1, which='LM')
            
            # Compute Dk
            Zk_tilde = -delta*np.outer(u_max,v_max)
            direction_vec = Zk_tilde.flatten() - Zk.flatten()
            
            # Compute new boundary
            grad = grad.toarray()
            wolfe_gap = grad.T.flatten() * direction_vec
            B_w = loss + wolfe_gap.sum()

            # Take the maximum bound
            if low_bound >= B_w:
                new_low_bound = low_bound
            else:
                new_low_bound = B_w

            # Compute Z_(k+1)
            alpha_k = 2/(it+2)
            Zk = (1-alpha_k)*Zk + alpha_k*Zk_tilde

            # Reset the sentinel back to the default value
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

        # Append results
        res_list.append(loss)
        ranks.append(rank_Z)
        
        if printing == True:
            if it % 10 == 0 or it == 1:
                print('Iteration:', it, 'Loss:', loss, 'Loss diff:', diff_loss, 'Rank(Z): ', rank_Z)

    # Print all the counters at the end of the algorithm
    print('Iteration:', it, 'Loss:', loss, 'Loss diff:', diff_loss, 'Rank(Z): ', rank_Z)
    print('Went to lower dim face:', B_used)
    print('Stayed:', A_used)
    print('Regular:', regularFW)
    print('Not entered if:', not_entered)

    print('in_border:', in_border)
    print('Inside ball:', inside)
    print('Outside ball:', outside)
    
    return Zk, loss, res_list, ranks
