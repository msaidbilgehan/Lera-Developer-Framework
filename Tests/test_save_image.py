
import libs

from image_tools import save_image, open_image

if __name__ == '__main__':
    image = open_image(r"C:\Users\said.bilgehan\Workspace\ASCII-Table.jpg")
    
    save_image([image], "", ["test_image"], "png")
