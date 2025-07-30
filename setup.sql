-- 电商客户服务场景模拟数据
CREATE OR REPLACE DATABASE cs_demo_db;
CREATE OR REPLACE SCHEMA cs_demo_db.data;
CREATE OR REPLACE WAREHOUSE cs_wh;

USE DATABASE cs_demo_db;
USE SCHEMA data;

CREATE OR REPLACE STAGE models 
    DIRECTORY = (ENABLE = TRUE);
    
-- 电商客户服务场景模拟数据

-- 1. 商品表
CREATE OR REPLACE TABLE products (
    product_id VARCHAR,
    product_name VARCHAR,
    category VARCHAR COMMENT '商品类型',
    price FLOAT COMMENT '商品价格',
    stock_quantity INTEGER,
    brand VARCHAR
);

INSERT INTO products VALUES
('P001', 'ZenPhone Pro 256GB', '智能手机', 8999.00, 50, 'TechZen'),
('P002', 'FlashMobile Ultra 512GB', '智能手机', 5999.00, 30, 'FlashTech'),
('P003', 'TabletMax Pro 12.9寸', '平板电脑', 8999.00, 100, 'TechZen'),
('P004', 'SmartPad Elite', '平板电脑', 4999.00, 80, 'DigitalWave'),
('P005', 'SoundBuds Pro 2', '蓝牙耳机', 1899.00, 25, 'TechZen'),
('P006', 'AudioMax WX-2000', '蓝牙耳机', 2399.00, 200, 'AudioMax'),
('P007', 'PowerBook Pro 14寸', '笔记本电脑', 15999.00, 20, 'TechZen'),
('P008', 'BusinessLaptop X1', '笔记本电脑', 12999.00, 15, 'ProTech');

-- 2. 订单表
CREATE OR REPLACE TABLE orders (
    order_id VARCHAR,
    customer_id VARCHAR,
    product_id VARCHAR,
    quantity INTEGER,
    total_amount FLOAT COMMENT '订单金额',
    order_status VARCHAR  COMMENT '订单的状态，包括正常状态如已完成，以及各种异常状态，如已换货、已退货等，以及中间状态，有已发货、换货中、退款中等',
    order_date TIMESTAMP,
    customer_name VARCHAR,
    customer_phone VARCHAR,
    shipping_address VARCHAR
);

INSERT INTO orders VALUES
('O001', 'C001', 'P001', 1, 8999.00, '已完成', '2024-01-15 10:30:00', '张三', '13800138001', '北京市朝阳区建国路1号'),
('O002', 'C002', 'P003', 2, 17998.00, '已取消', '2024-01-16 14:20:00', '李四', '13800138002', '上海市浦东新区陆家嘴路100号'),
('O003', 'C003', 'P005', 1, 1899.00, '已退货', '2024-01-18 09:15:00', '王五', '13800138003', '广州市天河区珠江路200号'),
('O004', 'C004', 'P007', 1, 15999.00, '已换货', '2024-01-20 16:45:00', '赵六', '13800138004', '深圳市南山区科技园300号'),
('O005', 'C005', 'P002', 1, 5999.00, '已发货', '2024-01-22 11:30:00', '钱七', '13800138005', '杭州市西湖区文三路400号'),
('O006', 'C006', 'P004', 1, 4999.00, '已完成', '2024-01-25 13:20:00', '孙八', '13800138006', '成都市锦江区春熙路500号'),
('O007', 'C007', 'P006', 1, 2399.00, '退款中', '2024-01-28 15:10:00', '周九', '13800138007', '武汉市武昌区中南路600号'),
('O008', 'C001', 'P008', 1, 12999.00, '已完成', '2024-02-01 08:45:00', '张三', '13800138001', '北京市朝阳区建国路1号'),
('O009', 'C008', 'P001', 1, 8999.00, '已发货', '2024-02-03 14:20:00', '吴十', '13800138008', '西安市雁塔区高新路700号'),
('O010', 'C009', 'P005', 2, 3798.00, '已完成', '2024-02-05 09:30:00', '郑十一', '13800138009', '南京市鼓楼区中山路800号'),
('O011', 'C010', 'P002', 1, 5999.00, '已退货', '2024-02-07 16:15:00', '王十二', '13800138010', '天津市和平区南京路900号'),
('O012', 'C011', 'P007', 1, 15999.00, '已完成', '2024-02-10 11:45:00', '李十三', '13800138011', '重庆市渝中区解放碑1000号'),
('O013', 'C012', 'P003', 1, 8999.00, '已换货', '2024-02-12 13:30:00', '张十四', '13800138012', '青岛市市南区香港路1100号'),
('O014', 'C013', 'P006', 2, 4798.00, '已完成', '2024-02-15 10:20:00', '陈十五', '13800138013', '大连市中山区人民路1200号'),
('O015', 'C014', 'P008', 1, 12999.00, '退款中', '2024-02-18 15:40:00', '刘十六', '13800138014', '苏州市姑苏区观前街1300号'),
('O016', 'C015', 'P004', 1, 4999.00, '已发货', '2024-02-20 08:50:00', '黄十七', '13800138015', '厦门市思明区中山路1400号');

-- 3. 物流表
CREATE OR REPLACE TABLE logistics (
    logistics_id VARCHAR,
    order_id VARCHAR,
    logistics_company VARCHAR,
    tracking_number VARCHAR,
    logistics_status VARCHAR,
    shipped_date TIMESTAMP,
    delivered_date TIMESTAMP,
    current_location VARCHAR
);

INSERT INTO logistics VALUES
('L001', 'O001', '顺丰速运', 'SF1234567890', '已签收', '2024-01-16 08:00:00', '2024-01-17 14:30:00', '北京市朝阳区'),
('L002', 'O002', '圆通速递', 'YT9876543210', '已取消', '2024-01-16 15:00:00', NULL, '上海分拣中心'),
('L003', 'O003', '中通快递', 'ZT5555666677', '退货运输中', '2024-01-19 10:00:00', NULL, '广州转运中心'),
('L004', 'O004', '韵达快递', 'YD1111222233', '换货运输中', '2024-01-21 09:30:00', NULL, '深圳分拣中心'),
('L005', 'O005', '申通快递', 'ST7777888899', '运输中', '2024-01-23 07:45:00', NULL, '杭州转运中心'),
('L006', 'O006', '顺丰速运', 'SF2468135790', '已签收', '2024-01-26 09:15:00', '2024-01-27 16:20:00', '成都市锦江区'),
('L007', 'O007', '京东物流', 'JD3691472580', '退货中', '2024-01-29 11:00:00', NULL, '武汉分拣中心'),
('L008', 'O008', '顺丰速运', 'SF9517534680', '已签收', '2024-02-02 08:30:00', '2024-02-03 15:45:00', '北京市朝阳区'),
('L009', 'O009', '中通快递', 'ZT8888999900', '运输中', '2024-02-04 10:15:00', NULL, '西安转运中心'),
('L010', 'O010', '韵达快递', 'YD2222333344', '已签收', '2024-02-06 08:20:00', '2024-02-07 16:30:00', '南京市鼓楼区'),
('L011', 'O011', '申通快递', 'ST4444555566', '退货运输中', '2024-02-08 14:45:00', NULL, '天津转运中心'),
('L012', 'O012', '顺丰速运', 'SF6666777788', '已签收', '2024-02-11 09:30:00', '2024-02-12 14:20:00', '重庆市渝中区'),
('L013', 'O013', '京东物流', 'JD9999000011', '换货运输中', '2024-02-13 11:15:00', NULL, '青岛分拣中心'),
('L014', 'O014', '圆通速递', 'YT1111222233', '已签收', '2024-02-16 08:40:00', '2024-02-17 15:50:00', '大连市中山区'),
('L015', 'O015', '中通快递', 'ZT3333444455', '退货中', '2024-02-19 13:25:00', NULL, '苏州分拣中心'),
('L016', 'O016', '韵达快递', 'YD5555666677', '运输中', '2024-02-21 07:30:00', NULL, '厦门转运中心');

-- 4. 商品评论表
CREATE OR REPLACE TABLE product_reviews (
    review_id VARCHAR,
    order_id VARCHAR,
    product_id VARCHAR,
    customer_id VARCHAR,
    rating INTEGER,
    review_content TEXT COMMENT '评论内容',
    review_date TIMESTAMP,
    is_verified_purchase BOOLEAN
);

INSERT INTO product_reviews VALUES
('R001', 'O001', 'P001', 'C001', 5, '手机很棒，拍照效果非常好，系统流畅，值得购买！', '2024-01-20 10:00:00', TRUE),
('R002', 'O006', 'P004', 'C006', 4, '平板屏幕清晰，性能不错，就是价格有点贵', '2024-01-30 14:30:00', TRUE),
('R003', 'O008', 'P008', 'C001', 5, '笔记本性能强劲，屏幕显示效果很好，办公游戏都很棒', '2024-02-05 09:15:00', TRUE),
('R004', 'O003', 'P005', 'C003', 2, '耳机连接不稳定，经常断连，音质也不如宣传的那么好，准备退货', '2024-01-25 16:20:00', TRUE),
('R005', 'O007', 'P006', 'C007', 1, '耳机质量有问题，用了两天就出现杂音，非常失望', '2024-02-02 11:45:00', TRUE),
('R006', 'O010', 'P005', 'C009', 5, 'SoundBuds音质很好，佩戴舒适，降噪效果不错，非常满意', '2024-02-10 14:20:00', TRUE),
('R007', 'O012', 'P007', 'C011', 4, 'PowerBook性能强大，做设计很流畅，就是发热有点大', '2024-02-15 16:30:00', TRUE),
('R008', 'O014', 'P006', 'C013', 5, 'AudioMax耳机音质一流，降噪效果很棒，强烈推荐！', '2024-02-20 11:15:00', TRUE),
('R009', 'O009', 'P001', 'C008', 3, '手机整体不错，但电池续航一般，一天需要充两次电', '2024-02-08 13:45:00', TRUE),
('R010', 'O011', 'P002', 'C010', 2, 'FlashMobile手机系统有时卡顿，相机效果也不如宣传的好', '2024-02-12 10:30:00', TRUE),
('R011', 'O013', 'P003', 'C012', 4, 'TabletMax屏幕质量很好，适合看视频和办公，就是存储空间有点小', '2024-02-17 15:20:00', TRUE),
('R012', 'O015', 'P008', 'C014', 1, '笔记本键盘有问题，多个按键失灵，影响正常使用', '2024-02-22 09:40:00', TRUE);

-- 5. 客服记录表
CREATE OR REPLACE TABLE customer_service_records (
    record_id VARCHAR,
    order_id VARCHAR,
    customer_id VARCHAR,
    service_type VARCHAR  COMMENT '客户申请的售后服务类型，如申请退货或换货、物流咨询等',
    issue_category VARCHAR COMMENT '客户申请的售后服务的原因，例如产品质量问题、配送问题等',
    conversation_log TEXT  COMMENT '客服沟通记录',
    service_agent VARCHAR,
    created_date TIMESTAMP,
    resolved_date TIMESTAMP,
    resolution_status VARCHAR COMMENT '客服解决的结果'
);

INSERT INTO customer_service_records VALUES
('CS001', 'O002', 'C002', '订单取消', '客户主动取消', 'Customer: 你好，我想取消订单O002，不需要这个商品了\nAgent: 您好，我来帮您处理订单取消。请稍等，我查看一下订单状态\nAgent: 您的订单还未发货，可以取消。我现在为您处理取消手续\nCustomer: 好的，谢谢。退款什么时候能到账？\nAgent: 退款会在1-3个工作日内原路返回到您的支付账户', '客服小王', '2024-01-16 15:30:00', '2024-01-16 16:00:00', '已解决'),
('CS002', 'O003', 'C003', '退货申请', '产品质量问题', 'Customer: 我买的SoundBuds Pro连接不稳定，经常断连，我要退货\nAgent: 您好，很抱歉给您带来不好的体验。请问具体是什么情况？\nCustomer: 和手机连接后经常自动断开，音质也不好\nAgent: 我理解您的困扰。我们支持7天无理由退货，我来为您办理退货手续\nCustomer: 需要我自己承担运费吗？\nAgent: 由于是产品质量问题，运费由我们承担。我安排快递员上门取件', '客服小李', '2024-01-22 10:15:00', '2024-01-23 14:30:00', '处理中'),
('CS003', 'O004', 'C004', '换货申请', '产品功能异常', 'Customer: 我的PowerBook键盘有几个键按不了，才买3天\nAgent: 您好，很抱歉出现这个问题。请问是哪几个按键？\nCustomer: 空格键和回车键经常失灵，影响正常使用\nAgent: 这确实影响使用体验。我为您申请换货，换一台全新的\nCustomer: 换货要多久？我工作急需用\nAgent: 我们会优先处理，明天安排取件，后天新机器就能送到', '客服小张', '2024-01-25 09:20:00', '2024-01-26 11:45:00', '已解决'),
('CS004', 'O007', 'C007', '退款申请', '产品质量问题', 'Customer: Sony耳机用了两天就出现杂音，我要退款\nAgent: 您好，请问具体是什么故障？\nCustomer: 左耳机有明显的电流声，右耳机声音很小\nAgent: 我来帮您处理。请问您还保留购买凭证和包装吗？\nCustomer: 都在，发票和盒子都没扔\nAgent: 好的，我现在为您登记退款申请，需要1-2个工作日审核', '客服小陈', '2024-02-03 13:40:00', NULL, '处理中'),
('CS005', 'O005', 'C005', '物流咨询', '配送延迟', 'Customer: 我的手机怎么还没到？已经3天了\nAgent: 您好，我来帮您查询物流信息\nAgent: 您的包裹目前在杭州转运中心，预计明天能送达\nCustomer: 为什么这么慢？\nAgent: 最近快递量比较大，给您造成不便很抱歉', '客服小刘', '2024-01-24 16:10:00', '2024-01-24 16:20:00', '已解决'),
('CS006', 'O003', 'C003', '退货跟进', '退货流程', 'Customer: 我的退货处理得怎么样了？\nAgent: 您好，我查看一下退货进度\nAgent: 您的退货商品我们已经收到，正在质检\nCustomer: 什么时候能退款？\nAgent: 质检通过后3-5个工作日内退款会到账', '客服小王', '2024-01-28 14:25:00', '2024-01-28 14:40:00', '已解决'),
('CS007', 'O004', 'C004', '换货跟进', '物流查询', 'Customer: 我的换货商品发了吗？\nAgent: 您好，我来查询一下\nAgent: 您的换货商品已经发出，快递单号是YD1111222233\nCustomer: 什么时候能到？\nAgent: 预计明天下午能送达，请保持电话畅通', '客服小李', '2024-01-30 10:30:00', '2024-01-30 10:45:00', '已解决'),
('CS008', 'O011', 'C010', '退货申请', '产品不满意', 'Customer: FlashMobile手机系统太卡了，我要退货\nAgent: 您好，请问具体是什么情况？\nCustomer: 打开应用很慢，经常卡死，和宣传不符\nAgent: 我理解您的困扰。请问您试过重启或恢复出厂设置吗？\nCustomer: 试过了，没有改善，我就要退货\nAgent: 好的，我为您办理退货手续，快递员明天上门取件', '客服小王', '2024-02-10 14:20:00', '2024-02-10 15:30:00', '已解决'),
('CS009', 'O013', 'C012', '换货申请', '屏幕问题', 'Customer: TabletMax屏幕有亮点，影响使用\nAgent: 您好，请问亮点在什么位置？大小如何？\nCustomer: 屏幕中央有个白点，很明显，看视频很影响\nAgent: 这确实影响使用体验。我为您申请换货，换一台全新的\nCustomer: 需要多久？\nAgent: 2-3个工作日，我们会优先处理', '客服小张', '2024-02-15 10:45:00', '2024-02-15 11:20:00', '已解决'),
('CS010', 'O015', 'C014', '退款申请', '产品缺陷', 'Customer: 笔记本键盘多个按键失灵，要求退款\nAgent: 您好，请问是哪些按键失灵？\nCustomer: W、A、S、D这几个按键都按不了，游戏都玩不了\nAgent: 这确实很影响使用。请问您是什么时候购买的？\nCustomer: 上周买的，才用了几天\nAgent: 我为您登记退款申请，由于是质量问题，全额退款', '客服小陈', '2024-02-20 16:30:00', NULL, '处理中'),
('CS011', 'O009', 'C008', '物流咨询', '配送延迟', 'Customer: 我的iPhone为什么还没到？\nAgent: 您好，我来帮您查询物流信息\nAgent: 您的包裹目前在西安转运中心，预计今天晚上能送达\nCustomer: 好的，谢谢', '客服小刘', '2024-02-06 14:15:00', '2024-02-06 14:25:00', '已解决'),
('CS012', 'O016', 'C015', '产品咨询', '使用问题', 'Customer: 华为平板怎么连接蓝牙耳机？\nAgent: 您好，我来教您连接步骤\nAgent: 请打开设置-蓝牙，确保蓝牙开启，然后点击搜索设备\nCustomer: 找到了，已经连接成功，谢谢\nAgent: 不客气，还有其他问题随时联系我们', '客服小李', '2024-02-22 11:20:00', '2024-02-22 11:35:00', '已解决'),
('CS013', 'O010', 'C009', '售后咨询', '保修问题', 'Customer: AirPods保修多久？如果坏了怎么办？\nAgent: 您好，AirPods享受1年全球联保\nAgent: 如果出现非人为损坏，可以免费维修或更换\nCustomer: 那人为损坏呢？\nAgent: 人为损坏需要支付维修费用，具体价格根据损坏程度而定', '客服小王', '2024-02-08 15:40:00', '2024-02-08 15:55:00', '已解决');

SELECT 'Data insertion completed successfully' as status;

-- 查询示例
select count(*) from customer_service_records;


SELECT orders.product_id, orders.order_id, product_reviews.review_id, product_name, brand, category, price, total_amount, rating, review_content, review_date, CONCAT('商品:', brand, ' ', category, ' ', product_name, '\n评论: ', review_content) as prod_review_content
    FROM products, orders, product_reviews 
    WHERE orders.product_id = products.product_id and orders.order_id = product_reviews.order_id
        and review_date >= '2024-01-01';

-- 在订单评论以及客服记录上创建Cortex Search Service
CREATE OR REPLACE CORTEX SEARCH SERVICE product_reviews_search
  ON prod_review_content
  ATTRIBUTES product_name, brand, category, price, total_amount, rating, review_content, review_date
  WAREHOUSE = cs_wh
  TARGET_LAG = '1 minute'
  AS (
    SELECT orders.product_id, orders.order_id, product_reviews.review_id, product_name, brand, category, price, total_amount, rating, review_content, review_date, CONCAT('商品:', brand, ' ', category, ' ', product_name, '\n评论: ', review_content) as prod_review_content
    FROM products, orders, product_reviews 
    WHERE orders.product_id = products.product_id and orders.order_id = product_reviews.order_id
        and review_date >= '2024-01-01'
);

SELECT PARSE_JSON(
    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
          'product_reviews_search',
          '{
             "query": "iPhone 15 Pro的评论怎么样?",
             "columns":[
                "product_name", "brand", "category", "price", "total_amount", "rating", "review_content", "review_date"
             ],
             "limit": 4
          }'
      )
  )['results'] as results;



SELECT orders.product_id, orders.order_id, customer_service_records.record_id, product_name, brand, category, price, total_amount, resolution_status, conversation_log, created_date, CONCAT('商品:', brand, ' ', category, ' ', product_name, '\n客服记录: ', conversation_log) as conversation_content
    FROM products, orders, customer_service_records 
    WHERE orders.product_id = products.product_id and orders.order_id = customer_service_records.order_id
        and created_date >= '2024-01-01';
        
CREATE OR REPLACE CORTEX SEARCH SERVICE cs_records_search
  ON conversation_content
  ATTRIBUTES product_name, brand, category, price, total_amount, resolution_status, conversation_log, created_date
  WAREHOUSE = cs_wh
  TARGET_LAG = '1 minute'
  AS (
    SELECT orders.product_id, orders.order_id, customer_service_records.record_id, product_name, brand, category, price, total_amount, resolution_status, conversation_log, created_date, CONCAT('商品:', brand, ' ', category, ' ', product_name, '\n客服记录: ', conversation_log) as conversation_content
    FROM products, orders, customer_service_records 
    WHERE orders.product_id = products.product_id and orders.order_id = customer_service_records.order_id
        and created_date >= '2024-01-01'
);


SELECT PARSE_JSON(
    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
          'cs_records_search',
          '{
             "query": "有哪些退货的记录?",
             "columns":[
                "product_name", "brand", "category", "price", "total_amount", "resolution_status", "conversation_log", "created_date"
             ],
             "limit": 4
          }'
      )
  )['results'] as results;
  