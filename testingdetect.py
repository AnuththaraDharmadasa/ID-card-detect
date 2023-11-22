import detect


detections = detect.run(weights='NICbest.pt',source= '0', save_crop='results',conf_thres=0.7 )