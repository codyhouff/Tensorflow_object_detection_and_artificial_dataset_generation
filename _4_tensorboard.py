import os
import threading
import webbrowser
from threading import Thread

###############################################

tensorboard_link = "http://LAPTOP-M6D3SOR6:6006/" # link to tensorboard, your personal link will be on the cmd prompt 

training_directory = "training/"                  # path to the folders where train iteration info goes

###############################################


#t = threading.Thread(target=os.system('tensorboard --logdir=' + training_directory), args=([]))
#t.start()
# tensorboard --logdir="training/training_results_3(old)/"
# webbrowser.open_new_tab(tensorboard_link)


def func1():
    t = threading.Thread(target=os.system('tensorboard --logdir=' + training_directory), args=([]))
    t.start()

def func2():
    webbrowser.open_new_tab(tensorboard_link)

if __name__ == '__main__':
    Thread(target = func1).start()
    Thread(target = func2).start()

print("1")
