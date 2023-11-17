anno_files = './Annotations_Part/%s.mat';
examples_path = './examples';
examples_imgs = dir([examples_path, '/', '*.jpg']);
cmap = VOClabelcolormap();

pimap = part2ind();     % part index mapping

for ii = 1:numel(examples_imgs)
    imname = examples_imgs(ii).name;
    img = imread([examples_path, '/', imname]);
    % load annotation -- anno
    load(sprintf(anno_files, imname(1:end-4)));
    
    [cls_mask, inst_mask, part_mask] = mat2map(anno, img, pimap);
    
    % display annotation
    subplot(2,2,1); imshow(img); title('Image');
    subplot(2,2,2); imshow(cls_mask, cmap); title('Class Mask');
    subplot(2,2,3); imshow(inst_mask, cmap); title('Instance Mask');
    subplot(2,2,4); imshow(part_mask, cmap); title('Part Mask');
    pause;
end
