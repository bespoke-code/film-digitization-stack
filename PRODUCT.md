# Scan to image

As a film photography enthusiast, 
I want to have a software that supports me from start to finish in stitching high-quality scans of my film shots 
And helps me with inverting photo negatives in order to get to the underlying photos.

Photos and images are terms used to interchangeably describe the files used as input, and output files as well.

Envisioned flow (for context): scan negatives with a digital camera, connect the SD card from the camera to a computer, 
copy the photos of the negatives to a folder, then open the folder (or subset of images from it) in this software.
Stitch images at the user's request, then batch-invert all images and adjust the levels for each channel (if color photo) 
or the grayscale channel to get the photos looking good from the get-go. User may adjust each image (curves and levels, 
exposure, contrast). User crops each photo if required. User saves photos in a format of choice offered by the software. 
Resulting (final) photos are saved as new files. Original files are kept unmodified on disk, previews are shown during 
edits and when viewing the image in general, but an "Original" button shows the original state of that photo.

The software supports RAW files from digital cameras, as well as JPEGs and PNGs as input.

The user starts the process of creating the final images by 
- opening a folder from the local filesystem, or
- selecting one or multiple files from a folder.

Between 2 and 16 images may be stitched together. Images being stitched must be color and perspective 
corrected/normalized before they are stitched and saved as one photo.

Some photo parts may be featureless/uniform (only film grain is visible on a monochromatic color). The user can 
be given the task to roughly arrange the parts of the final photo where no exact match can be found reliably. The final 
arranging and corrections are applied by the software automatically after confirmation of "rough arrangement".

Some negatives may be scanned as one photo in full, so no stitching is required.

The user must be able to crop the final image to size. Some borders or extra surface may be shown near the edge of the 
photos.

Every film has a uniform color base, which may be shown in photos. A pick tool must be available for the user to select 
the film base color. Auto-selection tool is useful as well, requiring user confirmation or override 
(using the pick tool).

This software inverts the image from a negative to a positive image at a click of a button. Multiple images, 
when selected, can be converted using the same button (batching).

The user can select images and mark them as monochrome. The software must treat them and convert the results to 
monochrome.

Any modification to the original photos is prohibited unless explicitly requested by the user by pressing a save button.

A histogram is available for the user in a right-side pane.
Curves and levels modifications can be adjusted at a click of the button (opens a new tool window). The image being 
worked on is previewed with the adjustments, and a preview checkbox can be unchecked to not preview the image.
