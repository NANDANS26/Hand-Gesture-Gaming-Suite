import pygame
import cv2
import mediapipe as mp
import numpy as np
import sys
import random
import time

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 60

# Colors
BACKGROUND = (15, 15, 25)
GRID_COLOR = (30, 30, 40)
SNAKE_HEAD = (50, 205, 50)
SNAKE_BODY = (34, 139, 34)
FOOD_COLOR = (220, 20, 60)
TEXT_COLOR = (240, 240, 240)
HIGHLIGHT = (255, 215, 0)
UI_BG = (20, 20, 30, 180)

# Initialize game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Controlled Snake")
clock = pygame.time.Clock()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Game variables
snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
snake_direction = (1, 0)
food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
score = 0
game_over = False
last_update_time = 0
update_interval = 0.15  # seconds
food_animation = 0
gesture_detected = False
current_gesture = "No hand detected"
sensitivity = 50  # pixels threshold for direction change
last_hand_detection_time = 0
hand_not_detected_timeout = 5  # seconds before showing warning

# Fonts
font_large = pygame.font.SysFont("Arial", 48, bold=True)
font_medium = pygame.font.SysFont("Arial", 32)
font_small = pygame.font.SysFont("Arial", 24)

# Shared variable for hand detection
hand_data = {
    "wrist_pos": None,
    "avg_finger_pos": None,
    "direction_vector": None,
    "finger_tips": [],
    "current_gesture": "No hand detected",
    "gesture_detected": False
}

def draw_grid():
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y), 1)

def draw_snake():
    for i, segment in enumerate(snake):
        color = SNAKE_HEAD if i == 0 else SNAKE_BODY
        rect = pygame.Rect(segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (color[0]//2, color[1]//2, color[2]//2), rect, 1)
        
        # Draw eyes on the head
        if i == 0:
            # Determine eye positions based on direction
            dx, dy = snake_direction
            if dx == 1:  # Right
                eye1 = (segment[0] * GRID_SIZE + GRID_SIZE - 5, segment[1] * GRID_SIZE + 7)
                eye2 = (segment[0] * GRID_SIZE + GRID_SIZE - 5, segment[1] * GRID_SIZE + GRID_SIZE - 7)
            elif dx == -1:  # Left
                eye1 = (segment[0] * GRID_SIZE + 5, segment[1] * GRID_SIZE + 7)
                eye2 = (segment[0] * GRID_SIZE + 5, segment[1] * GRID_SIZE + GRID_SIZE - 7)
            elif dy == 1:  # Down
                eye1 = (segment[0] * GRID_SIZE + 7, segment[1] * GRID_SIZE + GRID_SIZE - 5)
                eye2 = (segment[0] * GRID_SIZE + GRID_SIZE - 7, segment[1] * GRID_SIZE + GRID_SIZE - 5)
            else:  # Up
                eye1 = (segment[0] * GRID_SIZE + 7, segment[1] * GRID_SIZE + 5)
                eye2 = (segment[0] * GRID_SIZE + GRID_SIZE - 7, segment[1] * GRID_SIZE + 5)
                
            pygame.draw.circle(screen, (0, 0, 0), eye1, 3)
            pygame.draw.circle(screen, (0, 0, 0), eye2, 3)

def draw_food():
    global food_animation
    food_animation = (food_animation + 0.1) % (2 * np.pi)
    pulse = int(10 * np.sin(food_animation))
    
    center_x = food[0] * GRID_SIZE + GRID_SIZE // 2
    center_y = food[1] * GRID_SIZE + GRID_SIZE // 2
    
    # Draw pulsing food
    pygame.draw.circle(screen, FOOD_COLOR, (center_x, center_y), GRID_SIZE // 2 + pulse // 2)
    
    # Draw highlight
    highlight_pos = (center_x - GRID_SIZE // 4, center_y - GRID_SIZE // 4)
    pygame.draw.circle(screen, (255, 255, 255, 128), highlight_pos, GRID_SIZE // 6)

def draw_score():
    score_text = font_medium.render(f"Score: {score}", True, TEXT_COLOR)
    screen.blit(score_text, (20, 20))

def draw_gesture_info():
    # Draw semi-transparent background
    s = pygame.Surface((300, 140), pygame.SRCALPHA)
    s.fill(UI_BG)
    screen.blit(s, (WIDTH - 320, 20))
    
    # Draw gesture text
    gesture_text = font_small.render(f"Gesture: {current_gesture}", True, TEXT_COLOR)
    screen.blit(gesture_text, (WIDTH - 310, 40))
    
    # Draw sensitivity
    sens_text = font_small.render(f"Sensitivity: {sensitivity}", True, TEXT_COLOR)
    screen.blit(sens_text, (WIDTH - 310, 70))
    
    # Draw hand detection status
    hand_status = "Hand detected" if gesture_detected else "No hand detected"
    status_color = (50, 205, 50) if gesture_detected else (220, 20, 60)
    status_text = font_small.render(f"Status: {hand_status}", True, status_color)
    screen.blit(status_text, (WIDTH - 310, 100))
    
    # Draw instructions
    instr_text = font_small.render("Move fingers to control snake", True, TEXT_COLOR)
    screen.blit(instr_text, (WIDTH - 310, 130))

def draw_hand_warning():
    if not gesture_detected and time.time() - last_hand_detection_time > hand_not_detected_timeout:
        warning_text = font_medium.render("Show your hand to the camera!", True, (255, 165, 0))
        screen.blit(warning_text, (WIDTH // 2 - warning_text.get_width() // 2, HEIGHT - 50))

def draw_game_over():
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))
    
    game_over_text = font_large.render("GAME OVER", True, TEXT_COLOR)
    score_text = font_medium.render(f"Final Score: {score}", True, TEXT_COLOR)
    restart_text = font_small.render("Press SPACE to restart or ESC to quit", True, TEXT_COLOR)
    
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 60))
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2))
    screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 60))

def reset_game():
    global snake, snake_direction, food, score, game_over, update_interval
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    snake_direction = (1, 0)
    food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
    score = 0
    game_over = False
    update_interval = 0.15

def process_hand_gesture():
    global gesture_detected, current_gesture, snake_direction, last_hand_detection_time
    
    # Process webcam frame
    ret, frame = cap.read()
    if not ret:
        return True
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = hands.process(frame_rgb)
    
    # Reset hand data
    hand_data["wrist_pos"] = None
    hand_data["avg_finger_pos"] = None
    hand_data["direction_vector"] = None
    hand_data["finger_tips"] = []
    hand_data["current_gesture"] = "No hand detected"
    hand_data["gesture_detected"] = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get wrist and all finger tips coordinates
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            # Finger tips: index (8), middle (12), ring (16), pinky (20)
            finger_tips = [
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            ]
            
            # Convert to screen coordinates
            wrist_x = int(wrist.x * frame.shape[1])
            wrist_y = int(wrist.y * frame.shape[0])
            
            # Calculate average position of all finger tips
            avg_x = sum(int(tip.x * frame.shape[1]) for tip in finger_tips) // len(finger_tips)
            avg_y = sum(int(tip.y * frame.shape[0]) for tip in finger_tips) // len(finger_tips)
            
            # Calculate direction vector from wrist to average finger position
            dx = avg_x - wrist_x
            dy = avg_y - wrist_y
            
            # Determine direction based on displacement
            if abs(dx) > sensitivity or abs(dy) > sensitivity:
                if abs(dx) > abs(dy):
                    if dx > 0:
                        new_direction = (1, 0)  # Right
                        current_gesture = "RIGHT"
                    else:
                        new_direction = (-1, 0)  # Left
                        current_gesture = "LEFT"
                else:
                    if dy > 0:
                        new_direction = (0, 1)  # Down
                        current_gesture = "DOWN"
                    else:
                        new_direction = (0, -1)  # Up
                        current_gesture = "UP"
                
                # REMOVED: 180-degree turn prevention
                # Allow all direction changes, including opposite directions
                snake_direction = new_direction
                
                gesture_detected = True
                last_hand_detection_time = time.time()
            else:
                current_gesture = "Neutral"
                gesture_detected = True
                last_hand_detection_time = time.time()
            
            # Store hand data for visualization
            hand_data["wrist_pos"] = (wrist_x, wrist_y)
            hand_data["avg_finger_pos"] = (avg_x, avg_y)
            hand_data["direction_vector"] = (dx, dy)
            hand_data["finger_tips"] = finger_tips
            hand_data["current_gesture"] = current_gesture
            hand_data["gesture_detected"] = gesture_detected
            
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Draw direction vector and points
            if hand_data["wrist_pos"] and hand_data["avg_finger_pos"]:
                # Draw line from wrist to average finger position
                cv2.arrowedLine(frame, hand_data["wrist_pos"], hand_data["avg_finger_pos"], (0, 255, 0), 3)
                
                # Draw wrist point
                cv2.circle(frame, hand_data["wrist_pos"], 8, (255, 0, 0), -1)
                
                # Draw average finger position
                cv2.circle(frame, hand_data["avg_finger_pos"], 8, (0, 0, 255), -1)
                
                # Draw individual finger tips
                for tip in finger_tips:
                    tip_x = int(tip.x * frame.shape[1])
                    tip_y = int(tip.y * frame.shape[0])
                    cv2.circle(frame, (tip_x, tip_y), 6, (255, 255, 0), -1)
    
    # Display the camera feed in a separate window
    cv2.imshow('Hand Gesture Control', frame)
    
    # Check for ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        return False
    
    return True

def update_snake():
    global snake, food, score, game_over, update_interval
    
    # Calculate new head position
    head_x, head_y = snake[0]
    new_head = (head_x + snake_direction[0], head_y + snake_direction[1])
    
    # Check for collisions with walls
    if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
        new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
        game_over = True
        return
    
    # Check for collisions with self
    if new_head in snake[1:]:  # Skip the head when checking for collisions
        game_over = True
        return
    
    # Move snake
    snake.insert(0, new_head)
    
    # Check for food
    if new_head == food:
        score += 10
        # Increase speed slightly with score
        update_interval = max(0.08, update_interval - 0.002)
        
        # Generate new food position
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in snake:
                break
    else:
        # Remove tail if no food was eaten
        snake.pop()

# Main game loop
def main():
    global last_update_time, game_over, sensitivity, current_gesture, gesture_detected, last_hand_detection_time
    
    # Initialize hand detection time
    last_hand_detection_time = time.time()
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and game_over:
                    reset_game()
                elif event.key == pygame.K_UP:
                    sensitivity = min(100, sensitivity + 5)
                elif event.key == pygame.K_DOWN:
                    sensitivity = max(10, sensitivity - 5)
                # Debug: Keyboard controls
                elif event.key == pygame.K_w and not game_over:
                    snake_direction = (0, -1)
                elif event.key == pygame.K_s and not game_over:
                    snake_direction = (0, 1)
                elif event.key == pygame.K_a and not game_over:
                    snake_direction = (-1, 0)
                elif event.key == pygame.K_d and not game_over:
                    snake_direction = (1, 0)
        
        # Process hand gestures
        if not process_hand_gesture():
            running = False
        
        # Update current gesture from hand data
        current_gesture = hand_data["current_gesture"]
        gesture_detected = hand_data["gesture_detected"]
        
        # Update game state at fixed intervals
        current_time = time.time()
        if current_time - last_update_time > update_interval and not game_over:
            update_snake()
            last_update_time = current_time
        
        # Draw everything
        screen.fill(BACKGROUND)
        draw_grid()
        draw_food()
        draw_snake()
        draw_score()
        draw_gesture_info()
        draw_hand_warning()
        
        if game_over:
            draw_game_over()
        
        pygame.display.flip()
        clock.tick(FPS)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()