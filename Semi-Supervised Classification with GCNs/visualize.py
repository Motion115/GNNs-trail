from init import *
 
import matplotlib.pyplot as plt 

def plot(history: HistoryDict, title: str) -> None: 
    plt.suptitle(title, fontsize = 14) 
    # loss绘制
    ax1 = plt.subplot(121) 
    ax1.set_title("Loss") 
    ax1.plot(history["training_loss"  ], label="train") 
    ax1.plot(history["validation_loss"], label=  "val") 
    plt.xlabel("Epoch") 
    ax1.legend()
    # accuracy绘制
    ax2 = plt.subplot(122) 
    ax2.set_title("Accuracy") 
    ax2.plot(history["training_accuracy"  ], label="train") 
    ax2.plot(history["validation_accuracy"], label="val") 
    plt.xlabel("Epoch") 
    ax2.legend()
    # 更改此处以保存图片
    # plt.savefig(title,dpi=600)
    plt.show()