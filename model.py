from numpy import *

def compute_error_given_points(b, m, points):
    totalError = 0
    # for every point find the distance to the line
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        totalError += (y - (m * x + b)) ** 2
        
    # divide by length for average length (error)
    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    
    for i in range(num_iterations):
        # update b and m with newer and more accurate values (moving line closer to line of best fit) -> gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

# gradient = looks for smallest error 
def step_gradient(b_curr, m_curr, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    
    N = float(len(points))
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        # finding the direction with respect to b and m
        # computing partial derivatives of our error function 
        b_gradient += -(2 / N) * (y - ((m_curr * x) + b_curr))
        m_gradient += -(2 / N) * x * (y - ((m_curr * x) + b_curr))

    # update b & m values using partial derivatives
    
    new_b = b_curr - (learning_rate * b_gradient)
    new_m = m_curr - (learning_rate * m_gradient)
    return (new_b, new_m)
    
def run():
    # Collect the data and turn into an array of tuples using numpy library
    points = genfromtxt('data.csv', delimiter=',')
    
    # HyperParameters
    learning_rate = 0.0001 # defines how fast should the model converge
    
    # Slope Formula (y = mx + b)
    initial_b = 0
    initial_m = 0
    num_iterations=1000
    
    print ("starting gradient descent at b={0}, m={1}, error={2}".format(initial_b, initial_m, compute_error_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    
    
    print ("ending gradient descent at b={0}, m={1}, error={2}".format(num_iterations, b, compute_error_given_points(b, m, points)))

if __name__ == "__main__":
    run()