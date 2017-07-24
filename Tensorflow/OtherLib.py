from pylab import *

def main():
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    print(X)
    C,S = np.cos(X), np.sin(X)
    plot(X,C)
    plot(X,S)
    show()

if __name__ == '__main__':
    main()
