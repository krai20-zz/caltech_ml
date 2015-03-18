import matplotlib.pyplot as plt
import random
import numpy as np

def evaluate(w,x):
    """
    w: vector of weights
    x: vector of x values(x1, x2...xn)
    returns -1 if sign is negative and +1 if positive
    """
    total = sum([i*j for i, j in zip(w,x)])

    if total > 0:
        return 1
    elif total < 0:
        return -1

def create_points(x1,x2,y1,y2, num_points):
    """
    x1,x2,y1,y2: randomly chosen points
    num_points: number of points we choose to train
    """
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    points = []
    #choosing random points from X
    for i in xrange(1,num_points+1):
        # [artificial coordinate, x, y]
        point = [1, random.uniform(1,-1), random.uniform(1,-1)]
        random_x = point[1]
        random_y = point[2]

    #calculate value of y on line for the random value of x
        calculated_y = b+m*random_x

    # if randomly chosen value of y is on the right of the line, label is 1, else its 0
        if random_y > calculated_y:
            label = 1
        else:
            label = -1
        points.append((point,label))

    # plotting line and labels for training set
    x = [round(x1,4),round(x2,4)]
    y = [round(y1,4),round(y2,4)]
    plt.plot(x, y)

    for point, label in points:
        if label == -1:
            plt.plot(point[1], point[2],'ro')
        else:
            plt.plot(point[1], point[2],'bo')

    for xy in zip(x,y):
        plt.annotate(xy, xy=xy, textcoords="offset points")

    return points

def linear_regression(trained_points):

    #linear regression algorithm
    x_array = np.array([i[0] for i in trained_points])
    y_array = np.array([i[1] for i in trained_points])

    x_phi = np.linalg.pinv(x_array)
    w = np.dot(x_phi, y_array)

    return w

def error(w, points):

    #determining Ein (in sample error)
    misclassified_points = []

    #use trained_points for in sample error
    for point in points:

        if point[1] != evaluate(w, point[0]):

             misclassified_points.append(point)

    fraction_misclassified  = float(len(misclassified_points))/len(points)
    return fraction_misclassified

def perceptron(w, trained_points):

    iterations = 0
    iterate = True

    while iterate:
        points_misclassified = []
        iterations += 1
        iterate = False

        for point in trained_points:
            evaluated_label = evaluate(w,point[0])

            if evaluated_label != point[1]:
                points_misclassified.append(point)
                iterate = True

        if points_misclassified:
            random_misclassified = random.choice(points_misclassified)
            #learning algorithm
            w = map(lambda (w_original,x_original): w_original+random_misclassified[1]*x_original,
                                                    zip(w, random_misclassified[0]) )
    return iterations

if __name__ == '__main__':

    trials = 10
    insample_misclassified = 0
    outsample_misclassified = 0
    iterations_perceptron = 0

    for i in xrange(trials):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        y1 = random.uniform(-1,1)
        y2 = random.uniform(-1,1)

        trained_points = create_points(x1,x2,y1,y2,1000)
        w = linear_regression(trained_points)

        #find in sample error
        Ein_misclassified = error(w, trained_points)
        insample_misclassified += Ein_misclassified

        #generate 100 random out of sample points
        test_points = create_points(x1,x2,y1,y2,100)
        Eout_misclassified = error(w, test_points)
        outsample_misclassified += Eout_misclassified

        #find number of iterations for convergence of perceptron with inital w from linear regression
        iterations = perceptron(w, trained_points)
        iterations_perceptron += iterations

    print 'average insample prob of misclassification', insample_misclassified/trials
    print 'average out of sample prob of misclassification', outsample_misclassified/trials
    print 'average iterations for linear perceptron algorithm to converge', iterations_perceptron/trials
