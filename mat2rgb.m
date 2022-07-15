load(['XXX.mat']);
img=im2double(im);
img = min(max(img,0),1);
img_RGB=uint16(RSgenerate(img(:,:,[3 2 1]),1,1)*65535);
