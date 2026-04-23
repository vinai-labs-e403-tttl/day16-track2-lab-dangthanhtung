# Báo cáo Benchmark Mô hình Machine Learning (Phương án CPU)

## 1. Thông tin triển khai
- **Hạ tầng:** AWS EC2 - Instance Type: `r5.2xlarge` (8 vCPU, 32 GiB RAM).
- **Lý do sử dụng CPU:** Do hạn mức (quota) vCPU cho dòng Instance GPU (G/VT) trên tài khoản AWS mới bị giới hạn khởi điểm bằng 0. Để đảm bảo tiến độ bài Lab, phương án dự phòng sử dụng Instance CPU cấu hình cao (`r5.2xlarge`) đã được triển khai thay thế cho `g4dn.xlarge`.

- **Thông tin output của terraform:**

```
alb_dns_name = "ai-inference-alb-61c06abf-1341018532.us-east-1.elb.amazonaws.com"
bastion_public_ip = "13.223.89.84"
endpoint_url = "http://ai-inference-alb-61c06abf-1341018532.us-east-1.elb.amazonaws.com/v1/completions"
gpu_private_ip = "10.0.10.195"
```

## 2. Kết quả Benchmark (LightGBM)
Dưới đây là các chỉ số đo lường được trên bộ dữ liệu *Credit Card Fraud Detection*:

| Chỉ số | Kết quả |
|---|---|
| **Thời gian load dữ liệu** | ~1.75 giây |
| **Thời gian huấn luyện (Training)** | ~1.79 giây |
| **Độ chính xác (Accuracy)** | 99.79% |
| **AUC-ROC** | 0.7809 |
| **Độ trễ dự đoán (Inference Latency)** | ~0.37 ms / mẫu |
| **Thông lượng (Throughput)** | ~1.4 ms / 1000 mẫu |

## 3. Phân tích kết quả
- **Tốc độ huấn luyện:** Instance `r5.2xlarge` cho tốc độ xử lý rất ấn tượng với thuật toán LightGBM. Thời gian huấn luyện chưa đầy 2 giây cho thấy sức mạnh của 8 vCPU khi xử lý các bảng dữ liệu lớn.
- **Hiệu năng dự đoán:** Với độ trễ chỉ ~0.37ms per request, hệ thống hoàn toàn đáp ứng được các yêu cầu thực tế về kiểm tra gian lận thẻ tín dụng theo thời gian thực (Real-time inference).
- **Chỉ số Model:** Accuracy đạt mức rất cao (99.79%), tuy nhiên AUC-ROC (0.78) cho thấy mô hình vẫn có thể tối chỉnh thêm (fine-tune) để xử lý tốt hơn vấn đề mất cân bằng dữ liệu (imbalanced data) đặc trưng của bài toán Fraud Detection.

## 4. Kết luận
Mặc dù không sử dụng GPU, nhưng việc triển khai trên cấu hình CPU mạnh vẫn đảm bảo đầy đủ các bước trong quy trình Pipeline AI/ML trên Cloud: Từ khởi tạo hạ tầng bằng Terraform, thiết lập môi trường Python, đến huấn luyện và đo lường hiệu năng mô hình.
