% check the title above the image for how to use the user-interface to resize the input image
function im = seam_carve_image(im, sz)

% im = imresize(im, sz);
% 处理宽度调整（垂直接缝）
    delta_cols = size(im, 2) - sz(2);
    if delta_cols > 0
        im = remove_vertical_seams(im, delta_cols);
    elseif delta_cols < 0
        error('目标宽度大于当前宽度');
    end

    % 处理高度调整（水平接缝，通过转置处理）
    delta_rows = size(im, 1) - sz(1);
    if delta_rows > 0
        im = permute(im, [2 1 3]); % 转置图像，将高度转为宽度
        im = remove_vertical_seams(im, delta_rows); % 移除垂直接缝
        im = permute(im, [2 1 3]); % 转置回来
    elseif delta_rows < 0
        error('目标高度大于当前高度');
    end
end