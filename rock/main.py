import cv2
import mediapipe as mp
import random
import time
import numpy as np
import math

# --- Configuration and Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Use HD resolution for professional look
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Game Constants ---
GAME_STATES = {
    "WAITING": 0,
    "COUNTDOWN": 1,
    "SHOW_RESULT": 2,
    "ROUND_RESULT": 3,
    "MATCH_RESULT": 4
}

MOVES = ['Rock', 'Paper', 'Scissors']
MOVE_EMOJIS = {'Rock': '✊', 'Paper': '✋', 'Scissors': '✌️'}
MOVE_COLORS = {
    'Rock': (41, 128, 185),      # Blue
    'Paper': (39, 174, 96),      # Green  
    'Scissors': (192, 57, 43),   # Red
    'Waiting': (127, 140, 141)   # Gray
}
WINNING_SCORE = 2

# Landmark indices for finger tips and knuckles
TIP_ID = [4, 8, 12, 16, 20]
KNUCKLE_ID = [5, 9, 13, 17]

# --- Professional UI Colors ---
COLORS = {
    'background': (13, 17, 23),
    'panel': (22, 27, 34),
    'accent': (47, 129, 247),
    'success': (46, 204, 113),
    'warning': (241, 196, 15),
    'danger': (231, 76, 60),
    'text_primary': (248, 250, 252),
    'text_secondary': (158, 167, 180),
    'border': (48, 54, 61)
}

# --- Game Class ---
class RockPaperScissorsGame:
    def __init__(self):
        self.reset_game()
        self.game_state = GAME_STATES["WAITING"]
        self.countdown_start = 0
        self.result_display_start = 0
        self.round_result_display_start = 0
        self.computer_move_history = []
        self.animation_progress = 0
        
    def reset_game(self):
        self.score_player = 0
        self.score_computer = 0
        self.round_number = 1
        self.player_move = "Awaiting Gesture"
        self.computer_move = "???"
        self.round_winner = ""
        self.match_winner = ""
        self.countdown_value = 3
        self.computer_move_history = []
        self.animation_progress = 0
        
    def start_countdown(self):
        self.game_state = GAME_STATES["COUNTDOWN"]
        self.countdown_start = time.time()
        self.countdown_value = 3
        self.player_move = "Get Ready"
        self.computer_move = "???"
        self.animation_progress = 0
        
    def get_computer_move(self):
        """Enhanced computer move selection with adaptive randomness"""
        if not self.computer_move_history:
            return random.choice(MOVES)
            
        recent_moves = self.computer_move_history[-5:]
        move_counts = {move: recent_moves.count(move) for move in MOVES}
        weights = [1, 1, 1]
        
        if len(recent_moves) >= 3:
            most_common = max(move_counts, key=move_counts.get)
            if move_counts[most_common] >= len(recent_moves) * 0.6:
                if most_common == 'Rock':
                    weights = [1, 2, 1]
                elif most_common == 'Paper':
                    weights = [1, 1, 2]
                else:
                    weights = [2, 1, 1]
        
        random_factor = random.random() * 0.3
        weights = [w + random_factor for w in weights]
        
        return random.choices(MOVES, weights=weights)[0]
        
    def evaluate_round(self, player_gesture):
        if player_gesture not in MOVES:
            self.player_move = "Invalid Gesture"
            self.computer_move = "???"
            self.round_winner = "Show a clear gesture!"
            self.game_state = GAME_STATES["SHOW_RESULT"]
            self.result_display_start = time.time()
            return
            
        self.computer_move = self.get_computer_move()
        self.computer_move_history.append(self.computer_move)
        self.player_move = player_gesture
        self.animation_progress = 0
        
        if self.player_move == self.computer_move:
            self.round_winner = "TIE ROUND!"
        elif (self.player_move == 'Rock' and self.computer_move == 'Scissors') or \
             (self.player_move == 'Paper' and self.computer_move == 'Rock') or \
             (self.player_move == 'Scissors' and self.computer_move == 'Paper'):
            self.round_winner = "PLAYER WINS ROUND!"
            self.score_player += 1
        else:
            self.round_winner = "COMPUTER WINS ROUND!"
            self.score_computer += 1
            
        self.round_number += 1
        self.game_state = GAME_STATES["ROUND_RESULT"]
        self.round_result_display_start = time.time()
        
    def check_match_winner(self):
        if self.score_player >= WINNING_SCORE:
            self.match_winner = "VICTORY! PLAYER WINS!"
            return True
        elif self.score_computer >= WINNING_SCORE:
            self.match_winner = "DEFEAT! COMPUTER WINS!"
            return True
        return False

# --- Gesture Recognition Logic ---
def determine_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers_open = []
    
    fingers_open.append(lm[TIP_ID[1]].y < lm[KNUCKLE_ID[0]].y)
    fingers_open.append(lm[TIP_ID[2]].y < lm[KNUCKLE_ID[1]].y)
    fingers_open.append(lm[TIP_ID[3]].y < lm[KNUCKLE_ID[2]].y)
    fingers_open.append(lm[TIP_ID[4]].y < lm[KNUCKLE_ID[3]].y)

    num_open = sum(fingers_open)
    
    if num_open == 4:
        return 'Paper'
    elif num_open == 0:
        return 'Rock'
    elif num_open == 2 and fingers_open[0] and fingers_open[1] and not fingers_open[2] and not fingers_open[3]:
        return 'Scissors'
        
    return 'Waiting'

# --- Enhanced UI Drawing Functions ---
def draw_rounded_rect(image, pt1, pt2, color, corner_radius=15, alpha=1.0):
    """Draw a rounded rectangle with optional transparency"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    if alpha < 1.0:
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    else:
        # Draw main rectangle
        cv2.rectangle(image, (x1 + corner_radius, y1), (x2 - corner_radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + corner_radius), (x2, y2 - corner_radius), color, -1)
        
        # Draw corner circles
        cv2.circle(image, (x1 + corner_radius, y1 + corner_radius), corner_radius, color, -1)
        cv2.circle(image, (x2 - corner_radius, y1 + corner_radius), corner_radius, color, -1)
        cv2.circle(image, (x1 + corner_radius, y2 - corner_radius), corner_radius, color, -1)
        cv2.circle(image, (x2 - corner_radius, y2 - corner_radius), corner_radius, color, -1)

def draw_text(image, text, position, font_scale=1, color=COLORS['text_primary'], thickness=2, 
              font=cv2.FONT_HERSHEY_SIMPLEX, shadow=True):
    """Draw text with optional shadow effect"""
    if shadow:
        cv2.putText(image, text, (position[0] + 2, position[1] + 2), font, 
                   font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(image, text, position, font, font_scale, color, thickness)

def draw_centered_text(image, text, y_offset=0, font_scale=1, color=COLORS['text_primary'], 
                      thickness=2, shadow=True):
    """Draw centered text with shadow"""
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = (image.shape[1] - text_size[0]) // 2
    y = (image.shape[0] + text_size[1]) // 2 + y_offset
    draw_text(image, text, (x, y), font_scale, color, thickness, shadow=shadow)

def draw_animated_circle(image, center, radius, color, progress):
    """Draw an animated circle that fills based on progress (0-1)"""
    if progress >= 1:
        cv2.circle(image, center, radius, color, -1)
    else:
        # Draw arc based on progress
        start_angle = -90
        end_angle = start_angle + int(360 * progress)
        cv2.ellipse(image, center, (radius, radius), 0, start_angle, end_angle, color, 4)

def draw_move_card(image, x, y, width, height, move_type, move_name, is_player=True, animation_progress=1.0):
    """Draw an elegant card showing a move with animation"""
    # Card background
    color = MOVE_COLORS.get(move_type, COLORS['panel'])
    if move_type == 'Waiting':
        color = tuple(int(c * 0.7) for c in color)
    
    draw_rounded_rect(image, (x, y), (x + width, y + height), color, 20, 0.9)
    
    # Border
    border_color = COLORS['accent'] if is_player else COLORS['warning']
    cv2.rectangle(image, (x, y), (x + width, y + height), border_color, 2)
    
    # Animated circle background
    if animation_progress < 1.0:
        center = (x + width // 2, y + height // 2)
        circle_radius = min(width, height) // 3
        draw_animated_circle(image, center, circle_radius, border_color, animation_progress)
    
    # Move emoji
    emoji = MOVE_EMOJIS.get(move_type, '❓')
    emoji_scale = 2 if animation_progress >= 1.0 else 1.5
    emoji_y = y + height // 2 - 20
    
    if animation_progress >= 1.0:
        text_size = cv2.getTextSize(emoji, cv2.FONT_HERSHEY_SIMPLEX, emoji_scale, 3)[0]
        emoji_x = x + (width - text_size[0]) // 2
        draw_text(image, emoji, (emoji_x, emoji_y), emoji_scale, COLORS['text_primary'], 3)
    
    # Move name
    name_y = y + height - 30
    text_size = cv2.getTextSize(move_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    name_x = x + (width - text_size[0]) // 2
    draw_text(image, move_name, (name_x, name_y), 0.7, COLORS['text_primary'], 2)

def draw_score_bar(image, x, y, width, height, player_score, computer_score, max_score):
    """Draw an elegant score bar"""
    # Background
    draw_rounded_rect(image, (x, y), (x + width, y + height), COLORS['panel'], 10)
    
    # Calculate widths
    total_points = player_score + computer_score
    if total_points > 0:
        player_width = int((player_score / max_score) * width)
        computer_width = int((computer_score / max_score) * width)
    else:
        player_width = computer_width = 0
    
    # Player score fill
    if player_width > 0:
        draw_rounded_rect(image, (x, y), (x + player_width, y + height), COLORS['success'], 10)
    
    # Computer score fill
    if computer_width > 0:
        comp_x = x + width - computer_width
        draw_rounded_rect(image, (comp_x, y), (x + width, y + height), COLORS['danger'], 10)
    
    # Score text
    score_text = f"{player_score} - {computer_score}"
    text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    draw_text(image, score_text, (text_x, text_y), 1, COLORS['text_primary'], 2)

def draw_panel(image, game):
    """Draw a professional side panel"""
    panel_width = 380
    panel_height = image.shape[0]
    panel_x = image.shape[1] - panel_width
    
    # Main panel background with gradient
    for i in range(panel_height):
        alpha = i / panel_height
        color = tuple(int(c1 * (1 - alpha) + c2 * alpha) 
                     for c1, c2 in zip(COLORS['background'], COLORS['panel']))
        cv2.line(image, (panel_x, i), (image.shape[1], i), color, 1)
    
    # Panel border
    cv2.line(image, (panel_x, 0), (panel_x, panel_height), COLORS['border'], 2)
    
    # Header
    header_height = 120
    draw_rounded_rect(image, (panel_x, 0), (image.shape[1], header_height), COLORS['accent'], 0)
    
    # Title
    title_x = panel_x + 30
    draw_text(image, "GESTURE RPS", (title_x, 50), 1.3, COLORS['text_primary'], 3)
    draw_text(image, "PROFESSIONAL EDITION", (title_x, 85), 0.6, COLORS['text_primary'], 1)
    
    # Score section
    draw_text(image, "MATCH PROGRESS", (title_x, 150), 0.8, COLORS['text_secondary'], 2)
    draw_score_bar(image, title_x, 170, panel_width - 60, 30, game.score_player, game.score_computer, WINNING_SCORE)
    
    # Round info
    draw_text(image, f"ROUND {game.round_number}", (title_x, 220), 0.7, COLORS['text_secondary'], 1)
    
    # Current moves section
    draw_text(image, "CURRENT MOVES", (title_x, 270), 0.8, COLORS['text_secondary'], 2)
    
    # Player move card
    move_card_width = panel_width - 60
    move_card_height = 80
    draw_move_card(image, title_x, 290, move_card_width, move_card_height, 
                  game.player_move, f"YOU: {game.player_move}", True, game.animation_progress)
    
    # Computer move card
    draw_move_card(image, title_x, 390, move_card_width, move_card_height, 
                  game.computer_move, f"CPU: {game.computer_move}", False, game.animation_progress)
    
    # Game status
    status_y = 500
    draw_text(image, "STATUS", (title_x, status_y), 0.8, COLORS['text_secondary'], 2)
    
    status_text = ""
    status_color = COLORS['text_primary']
    
    if game.game_state == GAME_STATES["WAITING"]:
        status_text = "Ready to Start"
        status_color = COLORS['success']
    elif game.game_state == GAME_STATES["COUNTDOWN"]:
        status_text = f"Countdown: {game.countdown_value}"
        status_color = COLORS['warning']
    elif game.game_state == GAME_STATES["SHOW_RESULT"]:
        status_text = "Invalid Gesture"
        status_color = COLORS['danger']
    elif game.game_state == GAME_STATES["ROUND_RESULT"]:
        status_text = "Round Complete"
        status_color = COLORS['accent']
    else:
        status_text = "Match Finished"
        status_color = COLORS['danger']
    
    draw_text(image, status_text, (title_x, status_y + 30), 0.9, status_color, 2)
    
    # Controls section
    controls_y = 580
    draw_text(image, "CONTROLS", (title_x, controls_y), 0.8, COLORS['text_secondary'], 2)
    draw_text(image, "S - Start Game", (title_x, controls_y + 30), 0.6, COLORS['text_primary'], 1)
    draw_text(image, "R - Reset Game", (title_x, controls_y + 55), 0.6, COLORS['text_primary'], 1)
    draw_text(image, "Q - Quit", (title_x, controls_y + 80), 0.6, COLORS['text_primary'], 1)
    
    # Gesture guide
    guide_y = 670
    draw_text(image, "GESTURES", (title_x, guide_y), 0.8, COLORS['text_secondary'], 2)
    draw_text(image, "✊ Rock - Closed Fist", (title_x, guide_y + 25), 0.5, COLORS['text_primary'], 1)
    draw_text(image, "✋ Paper - Open Hand", (title_x, guide_y + 45), 0.5, COLORS['text_primary'], 1)
    draw_text(image, "✌️ Scissors - V Sign", (title_x, guide_y + 65), 0.5, COLORS['text_primary'], 1)

# --- Main Game Loop ---
game = RockPaperScissorsGame()
print("Starting Professional Hand Gesture RPS Game. Press 's' to start, 'r' to reset, 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Set elegant dark background
    image[:] = COLORS['background']
    
    # Process camera feed in a framed area
    cam_width, cam_height = 840, 560
    cam_x, cam_y = 40, 80
    
    # Camera frame background
    draw_rounded_rect(image, (cam_x-10, cam_y-10), (cam_x + cam_width + 10, cam_y + cam_height + 10), 
                     COLORS['panel'], 20, 0.8)
    
    # Get and process camera frame
    ret, camera_frame = cap.read()
    if ret:
        camera_frame = cv2.flip(camera_frame, 1)
        camera_frame = cv2.resize(camera_frame, (cam_width, cam_height))
        
        # Process for hand detection
        frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Convert back to BGR for display
        camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)
        
        current_player_gesture = 'Waiting'
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw elegant hand landmarks
                mp_drawing.draw_landmarks(
                    camera_frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLORS['accent'], thickness=3, circle_radius=3),
                    mp_drawing.DrawingSpec(color=COLORS['success'], thickness=2, circle_radius=2))
                
                current_player_gesture = determine_gesture(hand_landmarks)
        
        # Place camera frame on main image
        image[cam_y:cam_y + cam_height, cam_x:cam_x + cam_width] = camera_frame
    
    # --- Game State Management ---
    current_time = time.time()
    
    # Handle key presses
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and game.game_state == GAME_STATES["WAITING"]:
        game.start_countdown()
    elif key == ord('r'):
        game.reset_game()
        game.game_state = GAME_STATES["WAITING"]
    
    # Animation progress
    if game.game_state == GAME_STATES["COUNTDOWN"]:
        elapsed = current_time - game.countdown_start
        game.countdown_value = max(0, 3 - int(elapsed))
        game.animation_progress = min(1.0, elapsed / 3)
        
        if elapsed >= 3:
            game.evaluate_round(current_player_gesture)
    
    elif game.game_state == GAME_STATES["SHOW_RESULT"]:
        game.animation_progress = min(1.0, (current_time - game.result_display_start) / 2)
        if current_time - game.result_display_start >= 2:
            game.game_state = GAME_STATES["WAITING"]
    
    elif game.game_state == GAME_STATES["ROUND_RESULT"]:
        game.animation_progress = min(1.0, (current_time - game.round_result_display_start) / 3)
        if current_time - game.round_result_display_start >= 3:
            if game.check_match_winner():
                game.game_state = GAME_STATES["MATCH_RESULT"]
            else:
                game.start_countdown()
    
    elif game.game_state == GAME_STATES["MATCH_RESULT"]:
        if current_time - game.round_result_display_start >= 5:
            game.reset_game()
            game.game_state = GAME_STATES["WAITING"]
    
    # --- UI Rendering ---
    
    # Draw the professional panel
    draw_panel(image, game)
    
    # Main display content
    main_display_width = image.shape[1] - 380
    
    # Header
    draw_text(image, "HAND GESTURE RECOGNITION", (50, 40), 1.1, COLORS['text_primary'], 2)
    draw_text(image, f"Current Gesture: {current_player_gesture}", (50, 70), 0.8, COLORS['text_secondary'], 1)
    
    # Game state specific displays
    center_x = main_display_width // 2
    
    if game.game_state == GAME_STATES["WAITING"]:
        draw_centered_text(image, "READY TO PLAY", -100, 1.5, COLORS['accent'], 3)
        draw_centered_text(image, "Press S to start the match", -50, 0.9, COLORS['text_secondary'])
        
    elif game.game_state == GAME_STATES["COUNTDOWN"]:
        if game.countdown_value > 0:
            countdown_text = str(game.countdown_value)
            scale = 4 + (1 - game.animation_progress) * 2  # Scale animation
            alpha = 0.5 + game.animation_progress * 0.5  # Fade in
            color = tuple(int(c * alpha) for c in COLORS['warning'])
            draw_centered_text(image, countdown_text, -50, scale, color, 6)
        else:
            draw_centered_text(image, "GO!", -50, 4, COLORS['success'], 6)
        
    elif game.game_state == GAME_STATES["SHOW_RESULT"]:
        draw_centered_text(image, game.round_winner, -50, 1.3, COLORS['danger'], 3)
        
    elif game.game_state == GAME_STATES["ROUND_RESULT"]:
        draw_centered_text(image, game.round_winner, -80, 1.5, COLORS['accent'], 3)
        winner_color = COLORS['success'] if "PLAYER" in game.round_winner else COLORS['danger']
        draw_centered_text(image, f"{game.player_move} vs {game.computer_move}", -20, 1.0, winner_color, 2)
        
    elif game.game_state == GAME_STATES["MATCH_RESULT"]:
        draw_centered_text(image, game.match_winner, -80, 1.8, COLORS['danger'], 4)
        draw_centered_text(image, f"Final Score: {game.score_player} - {game.score_computer}", 0, 1.0, COLORS['text_primary'], 2)
        draw_centered_text(image, "Next match starting soon...", 50, 0.8, COLORS['text_secondary'])
    
    # Footer
    footer_y = image.shape[0] - 20
    draw_text(image, "Show your hand clearly in the camera view", (50, footer_y), 0.6, COLORS['text_secondary'])

    cv2.imshow('Professional Hand Gesture RPS', image)

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()