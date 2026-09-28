CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    signup_date DATE
);

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) REFERENCES customers(customer_id),
    status VARCHAR(30),
    total_amount NUMERIC(10, 2),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id VARCHAR(20) PRIMARY KEY,
    impressions BIGINT,
    clicks BIGINT,
    spend NUMERIC(10, 2)
);

INSERT INTO customers (customer_id, name, email, signup_date) VALUES
    ('CUST-001', 'Jane Rivera', 'jane.rivera@example.com', '2024-03-11'),
    ('CUST-002', 'Marcus Lee', 'marcus.lee@example.com', '2023-11-02')
ON CONFLICT (customer_id) DO NOTHING;

INSERT INTO orders (order_id, customer_id, status, total_amount) VALUES
    ('ORD-1029', 'CUST-001', 'shipped', 84.50),
    ('ORD-1030', 'CUST-001', 'processing', 240.00),
    ('ORD-1031', 'CUST-002', 'delivered', 59.99)
ON CONFLICT (order_id) DO NOTHING;

INSERT INTO campaigns (campaign_id, impressions, clicks, spend) VALUES
    ('CAMP-500', 1200000, 15400, 3200.00),
    ('CAMP-501', 800000, 9600, 2100.00)
ON CONFLICT (campaign_id) DO NOTHING;
