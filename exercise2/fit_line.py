import matplotlib.pyplot as plt
import numpy as np

def my_linfit(x, y):
    N = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    
    a = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / N
    
    return a, b

# Main program
def main():
    points = []

    def on_click(event):
        if event.button == 1:  # Left click: add point
            points.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'kx')
            plt.draw()
        elif event.button == 3:  # Right click: stop collecting
            plt.disconnect(cid)
            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points])
            a, b = my_linfit(x, y)
            xp = np.linspace(min(x) - 1, max(x) + 1, 100)
            plt.plot(xp, a * xp + b, 'r-')
            print(f"My fit: a={a} and b={b}")
            plt.show()

    fig, ax = plt.subplots()
    ax.set_title('Click to add points (Right-click to stop)')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__ == "__main__":
    main()