import matplotlib.pyplot as plt
import random

def evaluate(w,x):
    """
    w: vector of weights
    x: vector of x values(x1, x2...xn)
    returns -1 if sign is negative and +1 if positive
    """
    total = sum(i*j for i, j in zip(w,x))

    if total > 0:
        return 1
    elif total < 0:
        return -1

def train(x1,x2,y1,y2, num_points):
    """
    x1,x2,y1,y2: randomly chosen points
    num_points: number of points we choose to train
    """
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    trained_points = []
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
        trained_points.append((point,label))

    # plotting line and labels for training set
    x = [round(x1,4),round(x2,4)]
    y = [round(y1,4),round(y2,4)]
    plt.plot(x, y)

    for point, label in trained_points:
        if label == -1:
            plt.plot(point[1], point[2],'ro')
        else:
            plt.plot(point[1], point[2],'bo')

    for xy in zip(x,y):
        plt.annotate(xy, xy=xy, textcoords="offset points")

    #initialize weight vector to 0, so all points are misclassified initially
    w = [0,0,0]
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

    #determining probability of misclassification between g and f for 1000 random points
    misclassifications = 0

    for iteration in xrange(1000):
        point = [1, random.uniform(-1,1), random.uniform(-1,1)]

        if point[2] > b+m*point[1]:
            label = 1
        else:
            label = -1

        if label != evaluate(w, point):
            misclassifications += 1

    prob = float(misclassifications)/1000

    return iterations, prob

if __name__ == '__main__':

    trials = 100
    iterations_converge = 0
    prob = 0

    for i in xrange(trials):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        y1 = random.uniform(-1,1)
        y2 = random.uniform(-1,1)
        iterations, p = train(x1,x2,y1,y2,100)

        iterations_converge += iterations
        prob += p

    print 'average iterations for convergence', iterations_converge/trials
    print 'average probability of misclassification', prob/trials
