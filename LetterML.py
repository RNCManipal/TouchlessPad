import cv2
import keyboard
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

import execute_app as ex



def clear_whiteboard(display):
    """_summary_: Clears the whiteboard by drawing a white rectangle over it. """
    wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
    display[wb_y1-10:wb_y2+12, wb_x1-10:wb_x2+12] = (255, 255, 255)
    
    
def setup_display():
    """_summary_: Sets up the display window and the mouse callback. """
    title = np.zeros((80, 950, 3), dtype=np.uint8)
    board = np.zeros((600, 650, 3), dtype=np.uint8)
    panel = np.zeros((600, 300, 3), dtype=np.uint8)
    board[5:590, 8:645] = (255, 255, 255)
    
    board = cv2.rectangle(board, (8, 5), (645, 590), (255, 0, 0), 3)
    panel = cv2.rectangle(panel, (1, 4), (290, 590), (0, 255, 192), 2)
    panel = cv2.rectangle(panel, (22, 340), (268, 560), (255, 255, 255), 1)
    panel = cv2.rectangle(panel, (22, 65), (268, 280), (255, 255, 255), 1)
    
    cv2.line(panel, (145, 340), (145, 560), (255, 255, 255), 1)
    cv2.line(panel, (22, 380), (268, 380), (255, 255, 255), 1)
    
    cv2.putText(title, "       " +  window_name,(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(panel, "Action: ", (23, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(panel, "Best Predictions", (52, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(panel, "Prediction", (42, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, "Accuracy", (168, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, actions[1], (95, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_colors[actions[1]], 1)

    display = np.concatenate((board, panel), axis=1)
    display = np.concatenate((title, display), axis=0)
    
    return display


def setup_panel(display):
    """_summary_: Sets up the panel. """
    global execute_once
    action_region_pt1, action_region_pt2 = status_regions["action"]
    preview_region_pt1, preview_region_pt2 = status_regions["preview"]
    label_region_pt1, label_region_pt2 = status_regions["labels"]
    acc_region_pt1, acc_region_pt2 = status_regions["accs"]
    
    display[action_region_pt1[1]:action_region_pt2[1], action_region_pt1[0]:action_region_pt2[0]] = (0, 0, 0)
    display[preview_region_pt1[1]:preview_region_pt2[1], preview_region_pt1[0]:preview_region_pt2[0]] = (0, 0, 0)
    display[label_region_pt1[1]:label_region_pt2[1], label_region_pt1[0]:label_region_pt2[0]] = (0, 0, 0)
    display[acc_region_pt1[1]:acc_region_pt2[1], acc_region_pt1[0]:acc_region_pt2[0]] = (0, 0, 0)

    if crop_preview is not None:
        display[preview_region_pt1[1]:preview_region_pt2[1], preview_region_pt1[0]:preview_region_pt2[0]] = cv2.resize(crop_preview, (crop_preview_h, crop_preview_w)) 
    
    if best_predictions:
        labels = list(best_predictions.keys())
        accs = list(best_predictions.values())
        prediction_status_cordinate = [
            ((725, 505), (830, 505), (0, 0, 255)),
            ((725, 562), (830, 562), (0, 255, 0)),
            ((725, 619), (830, 619), (255, 0, 0))
        ]
        print("sp started with bp")
        
        if (execute_once == 0):
            ex.execute(labels[0])
            execute_once+=1
            
        print("Executed")
        
        for i in range(len(labels)):
            label_cordinate, acc_cordinate, color = prediction_status_cordinate[i]
            
            cv2.putText(display, labels[i], label_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, str(accs[i]), acc_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        for i in range(len(labels), 3):
            label_cordinate, acc_cordinate, color = prediction_status_cordinate[i]
            
            cv2.putText(display, "_", label_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, "_", acc_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        print("Label output")
    
    cv2.putText(display, "DRAW", (745, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_colors["DRAW"], 1)
    print("Sp done")


def mouse_click_event(event, x, y, flags, params):
    """_summary_: Mouse click event handler. """
    global crop
    if current_action is actions[1]:
        whiteboard_draw(event, x, y)
    elif current_action is actions[2] and crop == 0:
        crop = 1
        character_crop(event, x, y)
        
        
def whiteboard_draw(event, x, y):
    """_summary_: Draws on the whiteboard. """
    global left_button_down, right_button_down, maximums, prevx, prevy,crop
    
    wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
    if event is cv2.EVENT_LBUTTONUP:
        prevx, prevy = None, None
        left_button_down = False
    elif event is cv2.EVENT_RBUTTONUP:
        right_button_down = False
    elif wb_x1 <= x <= wb_x2 and wb_y1 <= y <= wb_y2:
        color = (0, 0, 0)
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP, cv2.EVENT_MOUSEMOVE]:
            if event is cv2.EVENT_LBUTTONDOWN:
                crop = 0
                color = (0, 0, 0)
                left_button_down = True
            elif left_button_down and event is cv2.EVENT_MOUSEMOVE:
                color = (0, 0, 0)
            elif event is cv2.EVENT_RBUTTONDOWN:
                color = (255, 255, 255)
                right_button_down = True
            elif right_button_down and event is cv2.EVENT_MOUSEMOVE:
                color = (255, 255, 255)
            else:
                return
            
            maximums.append((x, y))
            
            if prevx == None:
                cv2.circle(display, (x, y), 5, color, -1)
            else:
                cv2.line(display, (prevx, prevy), (x, y), color, 10)

            prevx, prevy = x, y
            cv2.imshow(window_name, display)
            
            
def character_crop(event, x, y):
    """_summary_: Crops the display automatically to include characters. """
    global bound_rect_cordinates, lbd_cordinate, lbu_cordinate, crop_preview, display, best_predictions, maximums
    
    print("cc started")
    high = maximums[0][1]
    low = maximums[0][1]
    right = maximums[0][0]
    left = maximums[0][1]
    for i in range(len(maximums)):
        if high>maximums[i][1]:
            high = maximums[i][1]
        if low<maximums[i][1]:
            low = maximums[i][1]
        if right<maximums[i][0]:
            right = maximums[i][0]
        if left>maximums[i][0]:
            left = maximums[i][0]
    
    high = high - 30
    low = low+ 30
    left = left- 30
    right = right + 30
    maximums = []
    
    print("maximum disabled")
    crop_preview = display[high:low, left:right].copy()
    crop_preview = np.invert(crop_preview)
    best_predictions = predict(model, crop_preview)
    display_copy = display.copy()
    high = low = right = left = 0
    setup_panel(display)
    print("setup updated")
    cv2.imshow(window_name, display)
    keyboard.press_and_release('e, d')
    print("reseted")
        
              
def load_model(path):
    """_summary_: Loads the Machine Learning Neural Network. """
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (5, 5), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(36, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.load_weights(path)
    
    return model


def predict(model, image):
    """_summary_: Predicts the character from the image. """
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
    prediction = model.predict(image)
    best_predictions = dict()
    
    for i in range(3):
        max_i = np.argmax(prediction[0])
        acc = round(prediction[0][max_i], 1)
        if acc > 0:
            label = labels[max_i]
            best_predictions[label] = acc
            prediction[0][max_i] = 0
        else:
            break
            
    return best_predictions


"""_summary_: Main function. """

"""_summary_: General putpose variable declaration. """
crop = 0
execute_once=0
maximums = []
prevx, prevy = None, None
left_button_down = False
right_button_down = False
bound_rect_cordinates = lbd_cordinate = lbu_cordinate = None
whiteboard_region = {"x": (20, 632), "y": (98, 656)}
window_name = "Live Cropped Character Recognition"
best_predictions = dict()
crop_preview_h, crop_preview_w = 238, 206
crop_preview = None
actions = ["N/A", "DRAW", "CROP"]
action_colors = {
    actions[0]: (0, 0, 255),
    actions[1]: (0, 255, 0),
    actions[2]: (0, 255, 192)
}
current_action = actions[0]
status_regions = {
    "action": ((736, 97), (828, 131)),
    "preview": ((676, 150), (914, 356)),
    "labels": ((678, 468), (790, 632)),
    "accs": ((801, 468), (913, 632))
}
model = load_model("Models/models/best_val_loss_model.h5")
display = setup_display()
cv2.imshow(window_name, display)
cv2.setMouseCallback(window_name, mouse_click_event)
pre_action = None
current_action = actions[1]


"""_summary_: Main loop. """
while True:
    k = cv2.waitKey(1)
    if k == ord('d') or k == ord('c'):
        if k == ord('d'):
            current_action = actions[1]
        elif k == ord('c'):
            execute_once=0
            current_action = actions[2]
        #if pre_action is not current_action:
        #    setup_panel(display)
        #    cv2.imshow(window_name, display)
        #    pre_action = current_action
    elif k == ord('e'):
        prevx, prevy = None, None
        clear_whiteboard(display)
        cv2.imshow(window_name, display)
    elif k == 27:
        break
        
cv2.destroyAllWindows()
