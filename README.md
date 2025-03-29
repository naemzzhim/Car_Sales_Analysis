# Car_Sales_Analysis
# 1. Tổng quan project
# 1.1. Bài toán
1. Phân tích tình hình bán xe theo Doanh thu, Khách hàng và Đại lý bán xe trong năm 2022 và 2023.
2. Dự đoán khả năng mua xe của Khách hàng dựa vào thu nhập hằng năm.
# 1.2. Quá trình thực hiện
* Giai đoạn 1: Giải quyết bài toán phân tích tình hình bán xe hơi trong hai năm 2022 và 2023, bộ dữ liệu được lấy trên Kaggle có tên là "Car_Sales_Report".
![image](https://github.com/user-attachments/assets/e33f0e3a-3a35-482c-b041-9e4acec38384)
File csv: 
![image](https://github.com/user-attachments/assets/f090b079-b92c-4497-b14e-9ed02f7a2e16)

 Vấn đề: Bộ dữ liệu chưa đầy đủ, type của ngày tháng còn bị sai, thiếu tên khách hàng, định dạng lại type của Đại lý. 

* Giai đoạn 2: Đặt ra các câu hỏi để từ đó tìm ra insight có liên quan để phân tích bài toán. Sau đó sẽ trực quan hóa dữ liệu bằng Matplotplb và Seaborn.

* Giai đoạn 3: Tiến hành giải quyết dự đoán khả năng có mua xe của khách hàng hay không bằng thuật toán học máy Logistic Regression, Random Forest Classification và SVC, sau đó đánh hiệu quả của từng thuật toán. 
# 2. Phân tích chi tiết bằng Python và Machine Learning
# 2.1. Bài toán 1:
# 2.1.1. Mô tả bộ dữ liệu
1. Kích thước dữ liệu:
* Số lượng dòng: 23,906.
* Số lượng cột: 16.
2. Các cột dữ liệu và loại dữ liệu:
* Car_id	(object):	Mã định danh duy nhất cho từng chiếc xe.
* Date	(object):	Ngày bán xe (định dạng chuỗi, có thể là ngày/tháng/năm).
* Customer Name	(object): Tên khách hàng mua xe.
* Gender	(object):	Giới tính của khách hàng (Male/Female).
* Annual Income	(int64): Thu nhập hàng năm của khách hàng (đơn vị tiền tệ không xác định).
* Dealer_Name	(object): Tên đại lý bán xe.
* Company	(object): Hãng xe (Ford, Toyota, Dodge, v.v.).
* Model	(object):	Mẫu xe.
* Engine	(object):	Loại động cơ (ví dụ: Overhead Camshaft).
* Transmission (object):	Loại hộp số (Auto hoặc Manual).
* Color	(object):	Màu sắc của xe.
* Price ($)	(int64): Giá bán của chiếc xe (đơn vị: USD).
* Dealer_No	(object):	Mã đại lý (có thể là mã vùng hoặc số nhận diện đại lý).
* Body Style (object):	Kiểu thân xe (SUV, Hatchback, Passenger, v.v.).
* Phone	(int64): Số điện thoại khách hàng.
* Dealer_Region	(object):	Khu vực đại lý (Middletown, Greenville, v.v.).
# 2.1.2. Làm sạch dữ liệu, xử lý dữ liệu khuyết thiếu và định dạng lại type dữ liệu
![image](https://github.com/user-attachments/assets/6fb0b622-9118-4c4c-a524-da8561498199)
![image](https://github.com/user-attachments/assets/d14dbd4e-f0e4-4a77-b8e5-397957ddb4b3)
![image](https://github.com/user-attachments/assets/8cc3d7e4-f5fa-4c3e-8e55-de4b8b1cfeed)
![image](https://github.com/user-attachments/assets/a33a694e-cc47-4c29-8b9e-dc6a009454a3)
# 2.1.3. Visualize bằng một số dashboard (EDA Analysis)
![image](https://github.com/user-attachments/assets/be18f5ac-8bcb-45b9-a3cf-187b35802f70)
![image](https://github.com/user-attachments/assets/20186fdd-68e9-4f95-9981-4d6da4b0807f)
![image](https://github.com/user-attachments/assets/1fea87e2-8a65-4027-89cc-98ffce6e8925)
![image](https://github.com/user-attachments/assets/49053a21-97ef-465f-9e5c-8523e9f78ed2)
![image](https://github.com/user-attachments/assets/11b5e492-3a1e-4307-a57e-5c53e82a8f92)
![image](https://github.com/user-attachments/assets/c9ff641e-c6f8-4678-96ab-cc0cbee05008)
![image](https://github.com/user-attachments/assets/edaff43b-f774-4cc1-811e-ac54181f79aa)
# 2.2. Bài toán 2:
Tiến hành dự đoán:
1. Tiền xử lí dữ liệu và tạo biến mục tiêu:
![image](https://github.com/user-attachments/assets/14160b2f-6903-4548-9b84-8c37ad3a4020)
2. Xử lý cân bằng nhãn ổn hay chưa:
![image](https://github.com/user-attachments/assets/d681c3bf-d03b-44e4-a6f2-b5b8dcea25f3)
![image](https://github.com/user-attachments/assets/7fce4aea-58b6-4dc1-9461-c22cdd1c9cc4)
3. Chia dữ liệu train 80, test 20:
![image](https://github.com/user-attachments/assets/398b121a-46f7-425a-986a-1959e4c57191)
4. Khởi tạo mô hình và huấn luyện:
* Random Forest Classification:
![image](https://github.com/user-attachments/assets/530d9d8a-17d8-48d0-9925-b78490d6a291)
![image](https://github.com/user-attachments/assets/0a299a18-2fa1-457c-8277-20acdcb71a46)

Khởi tạo lưới tham số và tìm tham số tốt nhất cho mô hình:
![image](https://github.com/user-attachments/assets/c15ffa79-e9b6-4aa5-b533-91698857e4a8)
![image](https://github.com/user-attachments/assets/f380f3b0-f925-48cc-8d89-040779f2f5d3)

* Logistic Regression:
![image](https://github.com/user-attachments/assets/0ea57942-3b19-44ac-b591-65a4b6ff1c36)
![image](https://github.com/user-attachments/assets/d359d5dc-da07-4b86-8725-3376437e3584)

Khởi tạo lưới tham số và tìm tham số tốt nhất cho mô hình:
![image](https://github.com/user-attachments/assets/5b046e91-362b-4d53-acf3-18b8f8b1eecb)
![image](https://github.com/user-attachments/assets/f6cc1e0d-6144-4f16-a1d9-6adc644bb7f4)

* SVC:
![image](https://github.com/user-attachments/assets/3a479c82-6dc3-492b-8c2e-d5d896690a38)
![image](https://github.com/user-attachments/assets/afc92087-cc2d-4252-a11a-98743027f269)

Khởi tạo lưới tham số và tìm tham số tốt nhất cho mô hình:
![image](https://github.com/user-attachments/assets/eab39309-e53b-4c9b-8d7c-5e2d77fb4c18)
![image](https://github.com/user-attachments/assets/d304771e-71fb-4be2-9ee6-034e352b2c3e)

* Qua đánh giá hiệu quả thì thấy rằng model Random Forest Classification hiệu quả nhất vì cho 4 chỉ số đánh giá đều ở mức tốt, Logistic Regression và SVC chưa thực sự ấn tượng bởi chỉ có Recall là cao còn 3 chỉ số còn lại Accuracy, Precision, F1 Score chưa tốt lắm, hơn nữa bộ dữ liệu này đã cân bằng bằng phương pháp SMOTE nên khi tìm tham số tốt nhất cũng sẽ chỉ cải thiện đôi chút hiệu quả của mô hình, vốn dĩ khi chưa tạo lưới tham số để tìm tham số tốt nhất thì model cũng đã mặc định chọn ra tham số là khả dĩ nhất để huấn luyện và cho ra kết quả đánh giá.

# 3. Visualize dataset bằng Power BI
Insight: Phân tích tình hình bán xe hơi trong hai năm 2022 và 2023 theo Doanh thu, Khách hàng và doanh số Đại lý.

Dashboard 1: Sales Overview
![image](https://github.com/user-attachments/assets/68e249eb-e922-4a27-9572-f31067df1be8)

Các chỉ số như: Doanh thu, Doanh số, Doanh thu theo Model xe, Brand, Khu vực và theo Ngày, Tháng, Năm.

Dashboard 2: Clients Analysis
![image](https://github.com/user-attachments/assets/07584cf2-3b1a-443c-b36d-86b79c08a310)

Các chỉ số như: Tổng số khách hàng, Tổng số khách hàng theo giới tính, Body Style của xe theo giới tính khách hàng, Doanh thu theo thu nhập của khách hàng, Table đầy đủ thông tin của khách hàng.

Dashboard 3: Performance Dealer
![image](https://github.com/user-attachments/assets/b926a98c-4c1c-4bd0-a5e9-9161a3790a49)

Các chỉ số như: Trung bình doanh thu, doanh số của một đại lý, xếp hạng tổng doanh thu, doanh số của các đại lý theo Ngày, Tháng, Năm. 

* Từ ba dashboard khi được visualize bằng Power BI ta sẽ thấy được chi tiết tình hình bán xe hơi, doanh thu, doanh số bao nhiêu, thời điểm nào bán nhiều, thời điểm nào bán ít, khách hàng thường mua xe kiểu gì và màu gì, khu vực nào bán được nhiều, khu vực nào bán được ít, từ những insight như vậy ta sẽ rút ra các decision phù hợp để cải thiện doanh thu và doanh số cho doanh nghiệp. 




 






























