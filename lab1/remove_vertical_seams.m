function im = remove_vertical_seams(im, num_seams)
    for s = 1:num_seams
        energy = costfunction(im);
        [h, w, ~] = size(im);
        
        % 动态规划计算最小累积能量（向量化）
        dp = zeros(h, w, 'single'); 
        from = zeros(h, w, 'int16'); 
        dp(1, :) = energy(1, :);
        
        for row = 2:h
            dp_prev = dp(row-1, :);
            % 构造左、中、右三个方向的能量数组
            left = [inf, dp_prev(1:end-1)];
            middle = dp_prev;
            right = [dp_prev(2:end), inf];
            
            % 取三个方向的最小值及对应索引
            [min_energy, dir] = min([left; middle; right], [], 1);
            from(row, :) = (1:w) + (dir - 2); % 计算来源列
            
            dp(row, :) = energy(row, :) + min_energy;
        end
        
        % 回溯接缝路径
        [~, col] = min(dp(end, :));
        seam = zeros(h, 1, 'int16');
        seam(end) = col;
        for row = h-1:-1:1
            col = from(row+1, col);
            seam(row) = col;
        end
        
        % 并行移除接缝
        new_im = zeros(h, w-1, size(im,3), class(im));
        parfor row = 1:h
            cols = [1:seam(row)-1, seam(row)+1:w];
            new_im(row, :, :) = im(row, cols, :);
        end
        im = new_im;
    end
end