
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import Cluster
from Cluster import Cluster
from init_centroids import init_centroids
from scipy.misc import imread

'''
if the user want to show the pic in the end
you need to put our of notes
the pixels array
the notes in '
'''

def compute_loss(clusters, size):
    sum = 0
    for cluster in clusters:
        sum += cluster.sum_loss()
    return sum/size


'''
the function runs the algorithm k_means
10 iterations
and compute the loss (in notes)
'''
def k_means (X,k):
    #print first iters
    print ("k="+str(k)+":")
    centroids = init_centroids(X,k)
    print("iter 0: " + print_cent(centroids))
    clusters = []
    # if you want to compute loss
    #loss = []
    '''
    #array that match pixel and index in clusters array.
    #if you want to show the pic - take out of notes
    pixels = []
    '''
    for clust in centroids:
        clusters.append(Cluster(clust))
    for iter in range(10):
        for pix in X:
            #distance
            min = np.linalg.norm(pix - clusters[0].getRGB())
            min_k = 0
            count = 0
            for cl in clusters:
                #find minimal distance
                dist = np.linalg.norm(pix - cl.getRGB())
                if dist < min:
                    min = dist
                    min_k = count
                count+=1
            #add pixels
            clusters[min_k].add_pixel()
            clusters[min_k].add_rgb(pix)

            #if you want to compute loss
            #clusters[min_k].add_to_loss(pix)

            '''
            #if you want to see pic - take out of notes
            if iter is 9:
                pixels.append(min_k)
            '''
        #if you want to see the loss
        #loss.append(compute_loss(clusters,len(X)))
        #update avg
        for c in clusters:
            c.update()
            c.clear_pix()
            '''
            #if you want to see the loss
            if iter < 9:
                c.clear_loss()
            '''
        #print the array of pixels
        clusters_rgb = []
        for rgb in clusters:
            clusters_rgb.append(rgb.getRGB())
        print("iter "+ str(iter+1) +": "+ print_cent(clusters_rgb))

    '''
    #if you want to see the loss
    plt.plot(loss)
    plt.title('K = %d' %k)
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.show()
    '''

    '''
    #if you want to show the pic - take out of notes
    for pix in range (0,len(X)):
        X[pix] = clusters[pixels[pix]].getRGB()
    '''

def print_cent(cent):
    if type(cent) == list:
       cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(
            ' ]', ']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(
            ' ]', ']').replace(' ', ', ')[1:-1]


def main():
    # data preperation (loading, normalizing, reshaping)
    path = 'dog.jpeg'
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])

    for i in [2,4,8,16]:
        k_means(X,i)

    '''
    #if you want to show the pic - take out of notes
    plt.imshow(A)
    plt.grid(False)
    plt.show()
    '''

if __name__ == "__main__":
    main()