# Schleimpilzmessung
This is a Program initially designed to analyze petri dishes of P. Polycephalum growing from the source to the nutrition medium.

## Usage
- Load a series of jpg-images from a folder using the "Open" button. The background is always the first image ordered by the python string comparison.
- The R1 to G1 buttons define the minimum Red, Blue and Green values for the pixels selected in the background subtracted from the image. Rm1 to Gm1 are the respective maxima. R2 to Bm2 are the values for the image subtracted from the background, useful on a bright foreground where the slime mold darkens the image.
- Enter the amount of pixels that are contained in a millimeter in the "Pixel/mm" field.
- The selection box initially labelled with "original" lets you select a view.
 - Original shows the image
 - Base shows the background
 - Diff shows whats recognized
 - Sub shows the subtraction of the image and the background added on top of each other
 - Contours shows the contours of the recognized area on top of the image. Red means that this area would be ignored / subtracted if inside a recognized (green) area.
- Zoom selects the magnification fo the image. Warning: Because TK canvas doesn't let you resize the image there, it gets resized before put in and requires memory accordingly.
- The selection box initially labelled with "last - first" lets you select the preview of a specific image to export.
- The "Export" button starts the export. This may take some time depending on the amount and size of the images.

All default values where good starting points for analyzing P. Polycephalum.

## Relevant buttons and keyboard shortcuts

- Draw rectangular areas to analyze using the left mouse button.
- Delete all rectangular areas at the cursor by double left clicking.
- Delete all areas by pressing "Ctrl+c" on the keyboard. Undo that by pressing "Ctrl-z".
- Print the pixel value of the image to the console using the right mouse button.

## Output
The export folder contains the recognized area on all images and a result.txt. The latter contains first some lines about the used configuration and later a tab separated table with the area (below the number), change in area, distance, change in distance, the top position relative to the marked area and the relative bottom position for each marking and for all images.
