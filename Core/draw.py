import tkinter.filedialog
import pygame
import sys
import json
import os
# from PyQt5.QtWidgets import QApplication, QFileDialog
# 上面是你原来的导入，如果只想用 PyQt 做文件对话框，可以保留。
# 如果你想用 tkinter 的对话框，可以改成:
import pygame.freetype

def run_pygame_window():
    # Initialize pygame
    pygame.init()

    # Set up display
    width, height = 810, 610
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Please draw your layer")

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    # Variables
    drawing = False
    points = []  # list to store points of the curve
    background_image = None

    # Set the frame rate
    FPS = 15
    clock = pygame.time.Clock()

    # Prepare the "Upload Image" button
    upload_button_text = "Upload Image"
    upload_button_font = pygame.freetype.Font(None, 24)
    upload_button_surface, upload_button_rect = upload_button_font.render(upload_button_text, BLACK)
    upload_button_rect.topleft = (20, 20)

    # Use a running flag instead of sys.exit()
    running = True

    while running:
        # ========== 事件处理 ==========
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # 当用户点击关闭按钮，只退出本循环和 Pygame，
                # 不要调用 sys.exit()。
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # 检查是否点在上传按钮上
                    if upload_button_rect.collidepoint(event.pos):
                        drawing = False

                        # ==== 方式 A: 用 PyQt 做文件对话框 ====
                        # from PyQt5.QtWidgets import QApplication, QFileDialog
                        #
                        # app = QApplication(sys.argv)
                        # file_dialog = QFileDialog.getOpenFileName()[0]
                        # if file_dialog:
                        #     background_image = pygame.image.load(file_dialog)
                        #     background_image = pygame.transform.scale(background_image, (width, height))

                        # ==== 方式 B: 用 tkinter 做文件对话框（更简单）====
                        root = tkinter.Tk()
                        root.withdraw()  # 不显示主窗体
                        file_path = tkinter.filedialog.askopenfilename()
                        root.destroy()

                        if file_path:  # 如果用户选择了文件
                            background_image = pygame.image.load(file_path)
                            background_image = pygame.transform.scale(background_image, (width, height))

                    else:
                        # 如果没有点按钮，就开始画线
                        drawing = True
                        points.append([])  # Start a new segment

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False

        # ========== 绘制逻辑 ==========
        if drawing:
            x, y = pygame.mouse.get_pos()
            if points:
                points[-1].append((x, y))

        # ========== 屏幕刷新 ==========
        window.fill(WHITE)

        # Draw the background image if available
        if background_image:
            window.blit(background_image, (0, 0))

        # Draw grid lines
        GRID_SIZE = 40  # Size of each grid cell
        TICK_SIZE = 10  # Size of the ticks on the axes
        LABEL_FONT = pygame.font.Font(None, 24)  # Font for the tick labels

        # for gx in range(0, width, GRID_SIZE):
        #     pygame.draw.line(window, BLACK, (gx, 0), (gx, height))
        # for gy in range(0, height, GRID_SIZE):
        #     pygame.draw.line(window, BLACK, (0, gy), (width, gy))
        
        for x in range(0, width, GRID_SIZE):
            pygame.draw.line(window, BLACK, (x, 0), (x, height))
            if x % (GRID_SIZE * 5) == 0:  # Draw longer tick marks at every 5 grid cells
                pygame.draw.line(window, BLACK, (x, -TICK_SIZE), (x, TICK_SIZE))
                label = LABEL_FONT.render(str(x), True, BLACK)  # Create label text
                window.blit(label, (x - label.get_width() // 2, TICK_SIZE))

        for y in range(0, height, GRID_SIZE):
            pygame.draw.line(window, BLACK, (0, y), (width, y))
            if y % (GRID_SIZE * 5) == 0:  # Draw longer tick marks at every 5 grid cells
                pygame.draw.line(window, BLACK, (-TICK_SIZE, y), (TICK_SIZE, y))
                label = LABEL_FONT.render(str(y), True, BLACK)  # Create label text
                window.blit(label, (TICK_SIZE, y - label.get_height() // 2))

        # Draw the upload button
        window.blit(upload_button_surface, upload_button_rect)

        # Draw the curve
        for segment in points:
            if len(segment) > 1:
                pygame.draw.lines(window, BLACK, False, segment, 6)

        pygame.display.flip()
        clock.tick(FPS)

    # ========== 退出循环后再做善后处理 ==========
    # Save points to JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "curve_points.json"), "w") as file:
        json.dump(points, file)

    pygame.quit()  # 只退出 Pygame，不退出整个解释器
    # 直接 return 即可，不要 sys.exit()
    return
