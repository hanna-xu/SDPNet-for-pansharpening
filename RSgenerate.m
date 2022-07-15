function [ image ] = RSgenerate( image,percent,colourization)
[hei,wid,channels] = size(image);
if colourization == 1
    for channel = 1:channels
    pixelvalues = sort(reshape(image(:,:,channel),[1,hei*wid]));
    top = pixelvalues(floor(hei*wid*(1-percent/100)));
    bottom = pixelvalues(max(ceil(hei*wid*percent/100),1));
    image(:,:,channel) = (image(:,:,channel)-bottom)/(top-bottom);
    end
else
    pixelvalues = sort(reshape(image,[1,hei*wid*channels]));
    top = pixelvalues(floor(hei*wid*channels*(1-percent/100)));
    bottom = pixelvalues(max(ceil(hei*wid*channels*percent/100),1));
    image = (image-bottom)/(top-bottom);
end
end

