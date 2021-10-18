"""
NAME
Authors: Anne-Sophie Ruh, Timon Meyer

DESCR.
"""

# python version 3.8.10 (libraries without pip name & version are default)
import cv2  # opencv-python 4.5.2.54
import json
import numpy as np  # numpy 1.17.4
import os
from PIL import Image, ImageTk  # Pillow 7.0.0
from tkinter import ttk, filedialog
import tkinter as tk

# combines the segments into larger ones if they overlap.
def combine_segments(segments):
    if len(segments) < 2:
        # 1 or 0 segments
        return segments

    res = set()
    for seg in segments:
        res = add_segment(seg, res)
    return res

# add a segment to an existing list of segments, combining them if they overlap.
def add_segment(seg, segments):
    used = set()
    new = list(seg)

    for i in segments:
        if i[0] <= new[0] <= i[1]:
            new[0] = i[0]
            used.add(i)
        if i[0] <= new[1] <= i[1]:
            new[1] = i[1]
            used.add(i)
        if new[0] <= i[0] <= i[1] <= new[1]:
            used.add(i)

    segments -= used
    segments.add(tuple(new))
    return segments

# recursively calculates the number of parents.
def parent_count(hierarchies, index):
    h = hierarchies[index][3]
    if h == -1:
        return 0
    return 1 + parent_count(hierarchies, h)

# Handling of the Mask rectangles
class MaskShard:
    def __init__(self, canvas: tk.Canvas, num, bbox):
        """creates a new mask shard from the canvas, its number and the bounding box"""
        self.l = min(bbox[0], bbox[2])
        self.t = min(bbox[1], bbox[3])
        self.r = max(bbox[0], bbox[2])
        self.b = max(bbox[1], bbox[3])
        self.canvas = canvas
        self.num = num

        # the references to the canvas objects, used for deleting and modification.
        self.r_handle = canvas.create_rectangle(self.l, self.t, self.r, self.b, outline="#f00")
        self.n_handle = canvas.create_text((self.l + self.r) / 2, (self.t + self.b) / 2, text=str(num), anchor="center", fill="#f00")

    def on_mouse(self, x, y):
        """Handles clicking the shard. Returns True if it survives. (self.shards = [i for i in self.shards if i.on_mouse(x, y)])"""
        if self.l <= x <= self.r and self.t <= y <= self.b:
            self.remove()
            return False
        return True

    def remove(self):
        # remove my parts from the canvas
        self.canvas.delete(self.r_handle, self.n_handle)

    def add(self, canvas):
        self.r_handle = canvas.create_rectangle(self.l, self.t, self.r, self.b, outline="#f00")
        self.n_handle = canvas.create_text((self.l + self.r) / 2, (self.t + self.b) / 2, text=str(self.num), anchor="center", fill="#f00")

    def serialize(self):
        # How it gets saved (as a list of its values).
        return [self.l, self.t, self.r, self.b, self.num]

    def deserialize(l, canvas):
        # How to read it from the saved state.
        return MaskShard(canvas, l[4], l[:4])

    def update_zoom(self, factor):
        # update my canvas representation on zoom.
        self.canvas.coords(self.n_handle, ((self.l + self.r) / 2 * factor, (self.t + self.b) / 2 * factor))
        self.canvas.coords(self.r_handle, (self.l*factor, self.t*factor, self.r*factor, self.b*factor))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # Initialize all values
        self.title("Test")
        self.path = None
        self.base_image = None
        self.preview_image = None
        self.img_handle = None
        self.rect_preview = None
        self.shards = []
        self.old_shards = None
        self.img = None

        self.rect = [0, 0, 0, 0]

        # setup User Interface (UI)
        self.create_ui()
        self.canvas.bind("<Button-1>", self.start_mask)
        self.canvas.bind("<Button-3>", self.print_mouse_color)
        self.canvas.bind("<ButtonRelease-1>", self.end_mask)
        self.canvas.bind("<B1-Motion>", self.mask_preview)
        self.canvas.bind("<Double-Button-1>", self.delete_mask)
        self.bind("<Control-c>", self.clear_mask)
        self.bind("<Control-z>", self.undo_mask)

    def create_diff_image(self, im1, im2):
        try:
            sub1 = cv2.subtract(im1, im2)
            sub2 = cv2.subtract(im2, im1)

            # TODO: another interesting structure, use maybe later?
            # lab = cv2.cvtColor(im1, cv2.COLOR_RGB2Lab)

            # extreme blue means condensing water vapor
            b, _1, _2 = cv2.split(sub1)
            sub1 = cv2.subtract(sub1, cv2.merge([b, b, b]))

            # green / red for vapor above white paper.
            _1, g, _2 = cv2.split(sub2)
            sub2 = cv2.subtract(sub2, cv2.merge([g, g, g]))

            # read the thresholds from the UI
            min_r1, min_g1, min_b1 = int(self.r1_sv.get()), int(self.g1_sv.get()), int(self.b1_sv.get())
            max_r1, max_g1, max_b1 = int(self.rm1_sv.get()), int(self.gm1_sv.get()), int(self.bm1_sv.get())
            min_r2, min_g2, min_b2 = int(self.r2_sv.get()), int(self.g2_sv.get()), int(self.b2_sv.get())
            max_r2, max_g2, max_b2 = int(self.rm2_sv.get()), int(self.gm2_sv.get()), int(self.bm2_sv.get())
            
            # binarize the image using the thresholds
            diff_image1 = cv2.inRange(sub1, np.array([min_b1, min_g1, min_r1]), np.array([max_b1, max_g1, max_r1]))
            diff_image2 = cv2.inRange(sub2, np.array([min_b2, min_g2, min_r2]), np.array([max_b2, max_g2, max_r2]))
            
            return (diff_image1, diff_image2, sub1, sub2) #, droplets)

        except ValueError:
            print(self.r1_sv.get(), self.g1_sv.get(), self.b1_sv.get(), self.rm1_sv.get(), self.gm1_sv.get(), self.bm1_sv.get())
            print(self.r2_sv.get(), self.g2_sv.get(), self.b2_sv.get(), self.rm2_sv.get(), self.gm2_sv.get(), self.bm2_sv.get())
            return None

    def create_preview(self, *args):
        # can't create a preview without files
        if self.base_image is not None and self.preview_image is not None:

            diffs = self.create_diff_image(self.preview_image, self.base_image)
            if diffs is None:
                print("Warning: any RGB value is not an integer")
                return
            else:
                # (diff_image1, diff_image2, sub1, sub2, droplets) = diffs
                (diff_image1, diff_image2, sub1, sub2) = diffs

            # cv2.imwrite("hsvtest.jpg", cv2.cvtColor(self.base_image, cv2.COLOR_BGR2HSV))

            mode = self.mode.get()
            if mode == "original":
                # shows the "original" image (the one subtracted from the base image, i.e the one with the P. Polycephalum)
                result = self.preview_image

            elif mode == "diff":
                # shows whats calculated as diff image.
                result = cv2.add(diff_image1, diff_image2)

            elif mode == "contours":
                # shows the contours as how they get exported

                diff = cv2.add(diff_image1, diff_image2)
                # use opencv to calculate the contours and hierarchies.
                contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                result = self.preview_image.copy()

                filtered, removed = [], []
                for (i, c) in enumerate(contours):
                    area = cv2.contourArea(c)
                    if area > 10 and parent_count(hierarchy[0], i) % 2 == 0:
                        # we only export large enough parts
                        filtered.append(c)
                    else:
                        # here we also show which tiny areas get removed.
                        removed.append(c)
                
                # draw the filtered areas green and the removed red.
                cv2.drawContours(result, filtered, -1, (0, 255, 0), 1)
                cv2.drawContours(result, removed, -1, (0, 0, 255), 1)
                # cv2.imwrite("contours.jpg", result)

            elif mode == "sub":
                result = 3 * cv2.add(sub1, sub2)
            elif mode == "base":
                result = self.base_image
            
            # Für HSV Falschfarben:
            # result = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

            # resize the image for zoom (TK and image scaling...)
            zoom = self.zoom.get()
            if zoom != "1":
                result = cv2.resize(result, (0, 0), fx=float(zoom), fy=float(zoom))

            # Update the UI
            self.img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)))

            if self.img_handle is None:
                self.img_handle = self.canvas.create_image(0, 0, anchor="nw", image=self.img)
            else:
                self.canvas.itemconfig(self.img_handle, image=self.img)

            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def get_files(self):
        # gives all images in the folder, sorted by their name (number in the beginning).
        return list(filter(lambda p: p.lower().endswith(".jpg"), sorted(os.listdir(self.path))))

    def open(self):
        """
        This opens the selected folder.
        """

        new_path = tk.filedialog.askdirectory()
        if new_path is None:
            return
        else:
            self.path = new_path

        for s in self.shards:
            s.remove()

        files = self.get_files()
        self.base_image = cv2.imread(os.path.join(self.path, files[0]))
        self.preview_image = cv2.imread(os.path.join(self.path, files[-1]))

        # some images are rotated by 0.2°. For this uncomment the following line
        # self.preview_image = cv2.warpAffine(self.preview_image, cv2.getRotationMatrix2D((self.preview_image.shape[1] // 2, self.preview_image.shape[0] // 2), -0.2, 1.), (self.preview_image.shape[1], self.preview_image.shape[0]))

        self.title(self.path)
        self.create_preview()

        # self.image_selection["values"] = tuple(["last - first"] + list(map(lambda x: x[1] + " - " + x[0], zip(files, files[1:]))))
        self.image_selection["values"] = list(map(lambda x: x + " - " + files[0], files[1:]))
        self.image_selection.current(len(files) - 2)

        if os.path.exists(os.path.join(self.path, "shards.json")):
            with open(os.path.join(self.path, "shards.json"), "r") as f:
                text = f.read()
                if text == "":
                    text = "[]"
                self.shards = [MaskShard.deserialize(i, self.canvas) for i in json.loads(text)]

    def export(self):
        files = self.get_files()

        exportpath = os.path.join(self.path, "export")
        # create export folder if it doesn't exist
        if not os.path.exists(exportpath):
            os.mkdir(exportpath)

        # backup the shards so we know what was used
        self.write_shards(exportpath)

        previous_image = cv2.imread(os.path.join(self.path, files[0]))

        # create header row of export
        res = [["Image"]]
        for i in self.shards:
            res[0] += [str(i.num), "𝚫area", "distance", "𝚫distance", "top", "bottom"]
        
        try:
            pxpmm = float(self.pxs.get())

            # now export each image
            for filename in files[1:]:
                print(filename)  # show where we are

                # read the image and create the difference images.
                img = cv2.imread(os.path.join(self.path, filename))
                diff = self.create_diff_image(img, previous_image)

                if diff is None:
                    # We have to abort an export if we can't generate difference images.
                    print("ABORTING: any RGB-value is not a number")
                    return
                else:
                    area_image = cv2.add(diff[0], diff[1])

                # write the current area image
                cv2.imwrite(os.path.join(exportpath, filename), area_image)

                line = [filename]
                for (i, s) in enumerate(self.shards):
                    # iterate over each shard and its index in the list.
                    cropped = area_image[s.t:s.b+1,s.l:s.r+1]

                    # let opencv calculate the contours and the hierarchy (if its a contour inside another etc.)
                    contours, hierarchy = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    filtered, removed = [], []
                    for (i2, c) in enumerate(contours):
                        area = cv2.contourArea(c)
                        # the number of parents indicates if its an area to add or to subtract as 0 means it gets added, 1 subtracted etc.
                        parents = parent_count(hierarchy[0], i2)

                        if area > 10 and parents % 2 == 0:
                            # parts with area <= 10 px usually are mistakes.
                            filtered.append(c)
                        elif parents % 2 == 1 and area > 10:
                            # but only subtract areas > 10 px (and not those)
                            removed.append(c)
                    
                    # area is the sum of all areas without the removed parts, changed into mm²
                    area = (
                        sum([cv2.contourArea(c) for c in filtered])
                        - sum([cv2.contourArea(c) for c in removed])
                    ) / (pxpmm ** 2)

                    segments = []
                    for c in filtered:
                        # the inner removed parts aren't interesting as we only care for the border.
                        _,y,_2,h = cv2.boundingRect(c)
                        segments.append((y, y+h))
                    
                    # combine_segments returns only bottom and top, we also want the length (both borders included!)
                    combined = [(i[1] - i[0] + 1, *i) for i in combine_segments(segments)]

                    dist, top, bottom = None, None, None
                    
                    # get the largest segment
                    for h in combined:
                        if dist is None or h[0] > dist:
                            (dist, top, bottom) = h
                    
                    # convert it to mm if it exists
                    if dist is None:
                        dist, top, bottom = 0, 0, 0
                    else:
                        dist /= pxpmm
                        top /= pxpmm
                        bottom /= pxpmm
                    
                    line += [
                        area,
                        # calculate the difference to before, 0 if it doesn't exist (res is the number of lines including the header line)
                        area - res[-1][4*i + 1] if len(res) > 1 else 0,
                        dist,
                        dist - res[-1][4*i + 3] if len(res) > 1 else 0,
                        top,
                        bottom
                    ]

                res.append(line)

        except ValueError:
            print("ABORTING: Either pixels/mm or any RGB-value is not a number")

        with open(os.path.join(exportpath, "result.txt"), "w") as f:
            # write the metadata
            f.write(f"pxpmm: {self.pxs.get()}\n")
            f.write(f"min_r1:\t{self.r1_sv.get()}\tmin_g1:\t{self.g1_sv.get()}\tmin_b1:\t{self.b1_sv.get()}\tmax_r1:\t{self.rm1_sv.get()}\tmax_g1:\t{self.gm1_sv.get()}\tmax_b1:\t{self.bm1_sv.get()}\n")
            f.write(f"min_r2:\t{self.r2_sv.get()}\tmin_g2:\t{self.g2_sv.get()}\tmin_b2:\t{self.b2_sv.get()}\tmax_r2:\t{self.rm2_sv.get()}\tmax_g2:\t{self.gm2_sv.get()}\tmax_b2:\t{self.bm2_sv.get()}\n\n")
            
            # write the export itself
            f.write("\n".join(["\t".join([str(j) for j in i]) for i in res]))

    def get_canvas_pos(self, x, y):
        # calculate the image position based on the UI position.
        (_, _, w, h) = self.canvas.bbox("all")
        (xs, xe) = self.canvas.xview() # from where to where we see, 0..1
        (ys, ye) = self.canvas.yview()
        xpos = (x - self.canvas.winfo_vrootx()) / self.canvas.winfo_width() * w * (xe-xs) + w * xs
        ypos = (y - self.canvas.winfo_vrooty()) / self.canvas.winfo_height() * h * (ye-ys) + h * ys
        return (int(xpos), int(ypos))

    def print_mouse_color(self, event, *args):
        # print the color at the position of the cursor
        (x, y) = self.get_canvas_pos(event.x, event.y)
        print(self.preview_image[y, x])

    def start_mask(self, event, *args):
        # start marking a mask shard.
        if self.preview_image is not None:
            (x, y) = self.get_canvas_pos(event.x, event.y)
            self.rect[0:2] = (x, y)
            self.rect_preview = self.canvas.create_rectangle(x, y, x, y, outline="#f00")

    def mask_preview(self, event, *args):
        # update the rectangle while drawing it.
        if self.rect_preview is not None:
            (x, y) = self.get_canvas_pos(event.x, event.y)
            self.canvas.coords(self.rect_preview, *self.rect[:2], x, y)

    def end_mask(self, event, *args):
        # doesn't make sense without base image.
        if self.preview_image is not None:
            self.canvas.delete(self.rect_preview)
            self.rect_preview = None

            self.rect[2:4] = self.get_canvas_pos(event.x, event.y)

            if abs(self.rect[0] - self.rect[2]) < 5 or abs(self.rect[1] - self.rect[3]) < 5:
                # don't create it if it's too small
                return

            n = 0
            i = 1
            while n == 0:
                if not any([j.num for j in self.shards if j.num == i]):
                    n = i
                    break
                i += 1

            zoom = float(self.zoom.get())
            # we need to rescale the bounding box. luckily (0, 0) stays the same
            self.shards.append(MaskShard(self.canvas, n, [int(i/zoom) for i in self.rect]))
            self.shards[-1].update_zoom(float(self.zoom.get()))
            # update shards file.
            self.write_shards(self.path)

    def write_shards(self, folder):
        # write the shards to the file.
        with open(os.path.join(folder, "shards.json"), "w") as f:
            json.dump([i.serialize() for i in self.shards], f)

    def delete_mask(self, event):
        # deletes all masks at the clicked position
        zoom = float(self.zoom.get())
        self.shards = [i for i in self.shards if i.on_mouse(*[i/zoom for i in self.get_canvas_pos(event.x, event.y)])]
        self.write_shards(self.path)

    def clear_mask(self, *args):
        # delete all shards but back them up
        self.old_shards = self.shards[:]
        for i in self.shards:
            i.remove()
        self.shards = []
        self.write_shards(self.path)

    def undo_mask(self, *args):
        # replace the mask with the backup if it exists
        if self.old_shards is not None:
            for i in self.shards:
                i.remove()
            self.shards = self.old_shards[:]
            for i in self.shards:
                i.add(self.canvas)
            self.old_shards = None
            self.write_shards(self.path)

    def switch_images(self, *args):
        # sets the base and preview image based on the selectbox.
        files = self.get_files()
        if len(files) < 2:
            return

        # this was used by an old version, but why not keeping it.
        if self.image_selection.get() == "last - first":
            f1 = files[0]
            f2 = files[-1]
        else:
            [f2, f1] = self.image_selection.get().split(" - ")

        self.base_image = cv2.imread(os.path.join(self.path, f1))
        self.preview_image = cv2.imread(os.path.join(self.path, f2))
        self.create_preview()

    def update_zoom(self, *args):
        # updates all shards on new zoom value.
        new = float(self.zoom.get())
        self.create_preview()
        for i in self.shards:
            i.update_zoom(new)


    def create_ui(self):
        """Handle all UI setup"""
        # The menu bar
        self.topframe = tk.Frame(self)
        self.topframe.pack(side="top")

        self.open_button = tk.Button(self.topframe, text="Open", command=self.open)
        self.open_button.grid(row=0, column=0, rowspan=2)

        # The threshold values
        self.r1_label = tk.Label(self.topframe, text="R1:")
        self.r1_label.grid(row=0, column=1)
        self.r1_sv = tk.StringVar(self, "10")
        self.r1_sv.trace("w", lambda *_: self.create_preview())
        self.r1_entry = tk.Entry(self.topframe, textvariable=self.r1_sv, width=3)
        self.r1_entry.grid(row=0, column=2)

        self.r2_label = tk.Label(self.topframe, text="R2:")
        self.r2_label.grid(row=1, column=1)
        self.r2_sv = tk.StringVar(self, "0")
        self.r2_sv.trace("w", lambda *_: self.create_preview())
        self.r2_entry = tk.Entry(self.topframe, textvariable=self.r2_sv, width=3)
        self.r2_entry.grid(row=1, column=2)

        self.g1_label = tk.Label(self.topframe, text="G1:")
        self.g1_label.grid(row=0, column=3)
        self.g1_sv = tk.StringVar(self, "10")
        self.g1_sv.trace("w", lambda *_: self.create_preview())
        self.g1_entry = tk.Entry(self.topframe, textvariable=self.g1_sv, width=3)
        self.g1_entry.grid(row=0, column=4)

        self.g2_label = tk.Label(self.topframe, text="G2:")
        self.g2_label.grid(row=1, column=3)
        self.g2_sv = tk.StringVar(self, "0")
        self.g2_sv.trace("w", lambda *_: self.create_preview())
        self.g2_entry = tk.Entry(self.topframe, textvariable=self.g2_sv, width=3)
        self.g2_entry.grid(row=1, column=4)

        self.b1_label = tk.Label(self.topframe, text="B1:")
        self.b1_label.grid(row=0, column=5)
        self.b1_sv = tk.StringVar(self, "0")
        self.b1_sv.trace("w", lambda *_: self.create_preview())
        self.b1_entry = tk.Entry(self.topframe, textvariable=self.b1_sv, width=3)
        self.b1_entry.grid(row=0, column=6)

        self.b2_label = tk.Label(self.topframe, text="B2:")
        self.b2_label.grid(row=1, column=5)
        self.b2_sv = tk.StringVar(self, "10")
        self.b2_sv.trace("w", lambda *_: self.create_preview())
        self.b2_entry = tk.Entry(self.topframe, textvariable=self.b2_sv, width=3)
        self.b2_entry.grid(row=1, column=6)

        self.rm1_label = tk.Label(self.topframe, text="Rm1:")
        self.rm1_label.grid(row=0, column=7)
        self.rm1_sv = tk.StringVar(self, "255")
        self.rm1_sv.trace("w", lambda *_: self.create_preview())
        self.rm1_entry = tk.Entry(self.topframe, textvariable=self.rm1_sv, width=3)
        self.rm1_entry.grid(row=0, column=8)

        self.rm2_label = tk.Label(self.topframe, text="Rm2:")
        self.rm2_label.grid(row=1, column=7)
        self.rm2_sv = tk.StringVar(self, "10")
        self.rm2_sv.trace("w", lambda *_: self.create_preview())
        self.rm2_entry = tk.Entry(self.topframe, textvariable=self.rm2_sv, width=3)
        self.rm2_entry.grid(row=1, column=8)

        self.gm1_label = tk.Label(self.topframe, text="Gm1:")
        self.gm1_label.grid(row=0, column=9)
        self.gm1_sv = tk.StringVar(self, "255")
        self.gm1_sv.trace("w", lambda *_: self.create_preview())
        self.gm1_entry = tk.Entry(self.topframe, textvariable=self.gm1_sv, width=3)
        self.gm1_entry.grid(row=0, column=10)

        self.gm2_label = tk.Label(self.topframe, text="Gm2:")
        self.gm2_label.grid(row=1, column=9)
        self.gm2_sv = tk.StringVar(self, "10")
        self.gm2_sv.trace("w", lambda *_: self.create_preview())
        self.gm2_entry = tk.Entry(self.topframe, textvariable=self.gm2_sv, width=3)
        self.gm2_entry.grid(row=1, column=10)

        self.bm1_label = tk.Label(self.topframe, text="Bm1:")
        self.bm1_label.grid(row=0, column=11)
        self.bm1_sv = tk.StringVar(self, "50")
        self.bm1_sv.trace("w", lambda *_: self.create_preview())
        self.bm1_entry = tk.Entry(self.topframe, textvariable=self.bm1_sv, width=3)
        self.bm1_entry.grid(row=0, column=12)

        self.bm2_label = tk.Label(self.topframe, text="Bm2:")
        self.bm2_label.grid(row=1, column=11)
        self.bm2_sv = tk.StringVar(self, "255")
        self.bm2_sv.trace("w", lambda *_: self.create_preview())
        self.bm2_entry = tk.Entry(self.topframe, textvariable=self.bm2_sv, width=3)
        self.bm2_entry.grid(row=1, column=12)

        self.pxs_label = tk.Label(self.topframe, text="Pixel/mm:")
        self.pxs_label.grid(row=0, column=13)
        self.pxs = tk.StringVar(self, "9.5")
        self.pxs_entry = tk.Entry(self.topframe, textvariable=self.pxs, width=4)
        self.pxs_entry.grid(row=0, column=14)

        # The view mode and other misc. UI parts.
        self.mode = ttk.Combobox(self.topframe, width=10)
        self.mode.state(["readonly"])
        self.mode["values"] = ["original", "base", "diff", "sub", "contours"]
        self.mode.current(0)
        self.mode.grid(row=1, column=13, columnspan=2)
        self.mode.bind('<<ComboboxSelected>>', self.create_preview)

        self.zoom_label = tk.Label(self.topframe, text="Zoom:")
        self.zoom_label.grid(row=0, column=15)

        self.zoom = ttk.Combobox(self.topframe, width=5)
        self.zoom.state(["readonly"])
        self.zoom["values"] = ["3", "2", "1.5", "1", "0.5", "0.25", "0.125"]
        self.zoom.current(3)
        self.zoom.grid(row=0, column=16)
        self.zoom.bind('<<ComboboxSelected>>', self.update_zoom)

        self.export_button = tk.Button(self.topframe, text="Export", command=self.export)
        self.export_button.grid(row=0, column=17)

        self.image_selection = ttk.Combobox(self.topframe, width=23)
        self.image_selection.state(["readonly"])
        self.image_selection["values"] = ["last - first"]
        self.image_selection.current(0)
        self.image_selection.grid(row=1, column=15, columnspan=3)
        self.image_selection.bind('<<ComboboxSelected>>', self.switch_images)

        # handle the scrollable canvas
        self.canvas = tk.Canvas(self)
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)


root = App()
root.mainloop() # Start the GUI