#Assignment-1
import numpy as np
from PIL import Image

def main():
    #QUESTION 1

    #question a
    arr = np.array([1, 2, 3, 6, 4, 5])
    arr = arr[::-1]
    print(arr)

    #question b
    array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
    array1 = array1.flatten()
    print(array1)

    #question c
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[1, 2], [3, 4]])
    if arr1.all() == arr2.all():
        print("True")

    #question d
    x = np.array([1,2,3,4,5,1,2,1,1,1])
    x1 = list(np.bincount(x))
    print(x1.index(max(x1)))
    print([index for index in range(len(x)) if x[index] == x1.index(max(x1))])
    y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3, ])
    x2 = list(np.bincount(y))
    print(x2.index(max(x2)))
    print([index for index in range(len(y)) if x[index] == x2.index(max(x2))])


    #question e
    gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')

    print("Sum of all elements:",np.sum(gfg))
    print("Sum of all elements:",np.sum(gfg))
    print("Sum of all rows:",np.sum(gfg,axis = 0))
    print("Sum of all columns:",np.sum(gfg,axis = 1))


    #question f
    n_array = np.array([[55, 25, 15],[30, 44, 2],[11, 45, 77]])

    print("Sum of diagonal:",np.sum(np.diag(n_array)))
    print("Eigen values and eigen vectors:",np.linalg.eig(n_array))
    print("Inverse of matrix:",np.linalg.inv(n_array))
    print("Determinant of matrix:",np.linalg.det(n_array))


    #question g
    p = [[1, 2], [2, 3]]
    q = [[4, 5], [6, 7]]
    print(np.matmul(p,q))
    print(np.cov(p,q))

    p = np.array([[1, 2], [2, 3], [4, 5]])
    q = np.array([[4, 5, 1], [6, 7, 2]])
    mul = np.matmul(p,q)
    print(mul)
    print(np.cov(mul))
    
    #question h 
    x = np.array([[2, 3, 4], [3, 2, 9]]); y = np.array([[1, 5, 0], [5, 10, 3]])
    print("Inner product:")
    print(np.inner(x,y))
    print("Outer product:")
    print(np.outer(x,y))
    print("Cross product:")
    print(np.cross(x,y))


    #QUESTION 2

    #question a
    array = np.array([[1,-2,3],[-4,5,-6]])
    temp = array.flatten()
    print(abs(array))
    print(temp[len(temp)//4],temp[len(temp)//2],temp[(len(temp)*3)//4])

    print("Mean:",np.mean(array))
    print("Median:",np.median(array))
    print("Standard deviation:",np.std(array))


    #question b
    a = np.array([-1.8,-1.6,-0.5,0.5,1.6,1.8,3.0])
    print("Floor values:\n",np.floor(a))
    print("Ceiling values:\n",np.ceil(a))
    print("Truncated values:\n",np.trunc(a))


    #QUESTION 3

    #question a

    array = np.array([10, 52, 62, 16, 16, 54, 453]);

    #Part i
    array.sort()
    print("Sorted array:")
    print(array)

    #Part ii
    indices = np.argsort(array)
    print("Indices of sorted array:")
    print(indices)

    #Part iii
    print("4 smallest elements:",array[0:4])

    #Part iv
    print("5 largest elements:",array[-5:len(array)])


    #question b

    array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])

    #Part i
    print("All integers:")
    print([i for i in array if i == int(i)])

    #Part ii
    print("All floating point numbers:")
    print([i for i in array if i != int(i)])


    #QUESTION 4

    #question a
    def img_to_array(path):
        img = Image.open(path)
        pixels = img.load()
        file = open("New.txt",'w')
        file.write(pixels)
        file.close()


if __name__ == "__main__":
    main()
    
