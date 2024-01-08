
% Compute new labels for each pixel, given current centroids
% @param image:     H x W x 3 matrix
% @param centroids: K x 3 matrix
% 
% @return:          H x W matrix
function [new_labels, sum_errors] = assign_clusters(image, centroids)
    num_clusters = size(centroids, 1);
    h = size(image, 1);
    w = size(image, 2);

    errors      = single(gpuArray(Inf(h, w)));
    new_labels  = single((ones(h, w)));

    for i = (1:num_clusters)

        % Create a H x W x 3 array of the color value of the current centroid
        cur_centroid = single(zeros(h, w, 3));
        for j = (1:h)
            for k = (1:w)
                cur_centroid(j, k, :) = centroids(i, :);
            end
        end
        
        cur_centroid = gpuArray(cur_centroid);
        % Calculate the squared differences between each pixel and the
        % centroid being examined
        dist_sq = (image - cur_centroid) .^ 2;
        sum_dist_sq = sqrt(sum(dist_sq, 3));

        % Calculate the difference between the best-known deviation
        % and the deviation from the current centroid
        error_diff = sum_dist_sq - errors;

        % Collapse error_diff so that error_diff indicates a 1 for each
        % pixel where the deviation from the current centroid is less
        % than the best known deviation

        % Update the labels
        error_diff = gather(error_diff);
        for j = (1:h)
            for k = (1:w)
                if error_diff(j, k) < 0
                    new_labels(j, k) = i;
                end
            end
        end

        % Update the best known errors
        errors = min(sum_dist_sq, errors);
    end

    % Move the new labels back to the GPU
    new_labels = gpuArray(new_labels);
    sum_errors = sum(errors, "all");
end

% Calculate each centroids by finding the mean color of all pixels
% assigned to the same label
function centroids = recalculate_centroids(image, labels, k)
    centroids   = single(gpuArray(zeros(k, 3)));
    % num_points  = single((ones(k, 1)));

    % image       = gather(image);
    % labels      = gather(labels);

    % for h = (1:size(labels, 1))
    %     for w = (1:size(labels, 2))
    %         cur_label = labels(h, w);
    % 
    %         centroids(cur_label, :) = centroids(cur_label, :) + squeeze(image(h, w, :)).';
    %         num_points(cur_label) = num_points(cur_label) + 1;
    %     end
    % end
    % 
    %  for i = (1:size(centroids, 1))
    %     centroids(i, :) = centroids(i, :) ./ num_points(i);
    % end
    k = gather(k);
    for i = (1:k)
        image_filter = (labels == i);
        filtered_image = image * image_filter;
        cur_centroid = sum(filtered_image, [1, 2]);
        num_points = sum(image, filter, "all") + 1;
        centroids(i, :) = cur_centroid / num_points;
    end
end

% Perform k-means 
% @param image:   Image to perform k-means clustering on pixels
% @param k:       Number of clusters
% @param caption: 
function kmc(image, k, caption, filename)
    image = gpuArray(image);
    % Select K centroids as a K x 3 matrix
    centroids = gpuArray(rand([k, 3], 'single'));

    % Create a H x W matrix of labels
    labels = single(gpuArray(zeros(size(image, 1), size(image, 2))));

    % Log the progression of errors over time
    objective_func_progression = [];

    while true
        centers_changed = false;

        % Assign each image to the nearest centroid,
        % given the centroids we currently have
        [new_labels, sum_errors] = assign_clusters(image, centroids);

        % Compare the current centroid assignment to the previous
        label_changes = new_labels - labels;

        for i = (1:size(new_labels, 1))
            for j = (1:size(new_labels, 2))
                if label_changes(i, j) ~= 0
                    centers_changed = true;
                    break;
                end
            end
            if centers_changed
                break
            end
        end

        fprintf("Total error = %f\n", sum_errors);
        objective_func_progression = [objective_func_progression, sum_errors];

        % Re-compute centroids
        labels = new_labels;
        centroids = recalculate_centroids(image, labels, k);

        % Show the image after each iteration
        clustered_image = single(gpuArray(zeros(size(image))));
        for h = (1:size(image, 1))
            for w = (1:size(image, 2))
                current_label = new_labels(h, w);
                clustered_image(h, w, :) = centroids(current_label, :);
            end
        end
        imshow(clustered_image);

        % Exit if no pixels were re-assigned to a new centroid
        if centers_changed == false
            break
        end
    end      

    % Show the final result
    caption = sprintf("K-means clustering with k=%d on %s", k, name);
    imshow(clustered_image);
    title(caption);

    % Plot the objective function results
    % plot_caption = sprintf("Total error on %s with k=%d, iter=%d", name, k, size(objective_func_progression, 2));
    % plot(objective_func_progression, "r.-");
    % title(plot_caption);
    size(objective_func_progression, 2)
end