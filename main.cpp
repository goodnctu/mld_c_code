#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <utility> // for std::pair
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>


std::vector<std::complex<double>> dotProduct(const std::vector<std::vector<std::complex<double>>>& R, const std::vector<std::complex<double>>& x) {
    size_t rows = R.size();
    size_t cols = x.size();
    std::vector<std::complex<double>> z(rows, std::complex<double>(0, 0));

    for (size_t i = 0; i < rows; ++i) {
        std::complex<double> total = 0.0;
        size_t j = 0;

        // Unroll the loop to process 4 elements at a time
        for (; j + 3 < cols; j += 4) {
            total += R[i][j] * x[j];
            total += R[i][j + 1] * x[j + 1];
            total += R[i][j + 2] * x[j + 2];
            total += R[i][j + 3] * x[j + 3];
        }

        // Handle remaining elements
        for (; j < cols; ++j) {
            total += R[i][j] * x[j];
        }

        z[i] = total;
    }
    return z;
}

struct ped_class {
    std::vector<std::complex<double>> cons;
    std::vector<std::vector<int>> dec_gray_r;
    std::vector<std::vector<int>> dec_gray_i;
    std::vector<int> dec_10_r;
    std::vector<int> dec_10_i;
    std::vector<double> dist;
    double total_dist = 0.0;
    std::vector<double> llr;
    std::vector<int> llr_hd;
};

std::pair<double, std::vector<int>> slicer(double r, double y, int layers);
std::pair<std::pair<double, double>, std::pair<std::vector<int>, std::vector<int>>> slicer_c(double r, const std::complex<double>& y_c, int layers);
double aabs(double in1, double in2);
std::complex<double> ic(std::complex<double> in1, std::complex<double> in2);
unsigned int bin2gray(unsigned int num);
unsigned int gray2bin(unsigned int num);
double bin2cons(int num, int max_num);
int bit2num(const std::vector<int>& binary_tuple);
std::vector<int> num2bit(int dec, int bit_num);
std::vector<int> generate_sequence(int size, int num_elements, int center_point);
std::vector<std::pair<int, int>> cand_gen(int size, int path_num, int x_init_idx_real, int x_init_idx_imag);

// slicer
inline std::pair<double, std::vector<int>> slicer(double r, double y, int layers) {
    std::vector<int> dec(layers + 1, 0);

    dec[0] = (y > 0)? 1 : 0;
    y = std::abs(y);
    std::vector<int> decision(layers, 0);
    double threshold = r * (1 << layers); 
    for (int i = 0; i < layers; ++i) {
        if (y > threshold) {
            y -= threshold;
            decision[i] = 1;
        } else {
            decision[i] = 0;
        }
        if (i == 0) {
            dec[1] = (decision[0])? 0 : 1;
        } else {
            dec[i+1] = decision[i-1] != decision[i]; // XOR operation in C++
        }
        threshold /= 2;
    }
    double ped = std::abs(y - r);
    return {ped, dec};
}

// slicer_c
std::pair<std::pair<double, double>, std::pair<std::vector<int>, std::vector<int>>> slicer_c(double r, const std::complex<double>& y_c, int layers) {
    auto [ped_r, dec_r] = slicer(r, y_c.real(), layers);
    auto [ped_i, dec_i] = slicer(r, y_c.imag(), layers);

    return std::make_pair(std::make_pair(ped_r, ped_i), std::make_pair(dec_r, dec_i));
}


inline double aabs(double in1, double in2) {
    return std::sqrt(in1 * in1 + in2 * in2);
}

inline std::complex<double> ic(std::complex<double> in1, std::complex<double> in2) {
    return in1 * in2; // 直接返回两个复数相乘的结果
}

unsigned int bin2gray(unsigned int num) {
    return num ^ (num >> 1);
}

inline unsigned int gray2bin(unsigned int num) {
    for (unsigned int mask = num >> 1; mask != 0; mask = mask >> 1) {
        num = num ^ mask;
    }
    return num;
}

inline double bin2cons(int num, int max_num) {
    return static_cast<double>(num) * 2 - max_num;
}

inline int bit2num(const std::vector<int>& binary_tuple) {
    int num = 0;
    for (size_t i = 0; i < binary_tuple.size(); ++i) {
        num = (num << 1) | binary_tuple[i];
    }
    return num;
}

inline std::vector<int> num2bit(int dec, int bit_num) {
    std::vector<int> binary_list(bit_num);
    for (int i = bit_num - 1; i >= 0; --i) {
        binary_list[i] = dec & 1;
        dec >>= 1;
    }
    return binary_list;
}


std::vector<int> generate_sequence(int size, int num_elements, int center_point) {
    std::vector<int> sequence;
    int start_point = center_point - (num_elements / 2);
    int end_point = start_point + num_elements;

    if (start_point < 0) {
        end_point -= start_point;
        start_point = 0;
    }
    if (end_point > size) {
        start_point -= (end_point - size);
        end_point = size;
        if (start_point < 0) {
            start_point = 0;
        }
    }

    for (int i = start_point; i < end_point; ++i) {
        sequence.push_back(i);
    }

    return sequence;
}

std::vector<std::pair<int, int>> cand_gen(int size, int path_num, int x_init_idx_real, int x_init_idx_imag) {
    int path_num_sqrt = static_cast<int>(std::sqrt(path_num));
    std::vector<int> real_cand = generate_sequence(size, path_num_sqrt, x_init_idx_real);
    std::vector<int> imag_cand = generate_sequence(size, path_num_sqrt, x_init_idx_imag);
    std::vector<std::pair<int, int>> cand;
    cand.reserve(real_cand.size() * imag_cand.size());
    for (int real : real_cand) {
        for (int imag : imag_cand) {
            cand.emplace_back(real, imag);
        }
    }

    return cand;
}

std::pair<std::vector<double>, std::vector<int>> run_hsinde(const std::vector<std::vector<std::complex<double>>>& R_upper, const std::vector<std::complex<double>>& z_in, int n_ss, int qam_bit, int path_num) {
    // 注意：这里需要根据实际情况调整类型，可能需要使用 std::vector<std::vector<std::complex<double>>>
    // 对 R_upper 和 z_in 进行处理的代码
    
    // 以下为示例代码片段，展示如何初始化 ped 结构
    int qam_bit_per_axis = qam_bit / 2;
    int slicer_layer = qam_bit_per_axis - 1;
    int points_per_axis = 1 << qam_bit_per_axis;
    int max_cons = points_per_axis - 1;

    std::vector<ped_class> ped(path_num);
    for (auto& p : ped) {
        p.cons.resize(n_ss, std::complex<double>(0, 0));
        p.dec_gray_r = std::vector<std::vector<int>>(n_ss, std::vector<int>(qam_bit_per_axis, 0));
        p.dec_gray_i = std::vector<std::vector<int>>(n_ss, std::vector<int>(qam_bit_per_axis, 0));
        p.dec_10_r.resize(n_ss, 0);
        p.dec_10_i.resize(n_ss, 0);
        p.dist.resize(n_ss, 0.0);
        p.total_dist = 0.0;
        p.llr.resize(qam_bit * n_ss, 0.0);
        p.llr_hd.resize(qam_bit * n_ss, 0);
    }
    
    // init_gen
    auto [init_ped, init_cons] = slicer_c(R_upper[0][0].real(), z_in[0], slicer_layer);


    int x_init_idx_real = gray2bin(bit2num(init_cons.first));
    int x_init_idx_imag = gray2bin(bit2num(init_cons.second));

    // cand_gen
    auto cand = cand_gen(points_per_axis, path_num, x_init_idx_real, x_init_idx_imag);



    for (int path_idx = 0; path_idx < path_num; ++path_idx) {
        ped[path_idx].dec_10_r[0] = cand[path_idx].first;
        ped[path_idx].dec_10_i[0] = cand[path_idx].second;

        // 注意：C++ 中没有直接的复数乘法运算符来乘以整数，因此需要创建复数
        ped[path_idx].cons[0] = std::complex<double>(bin2cons(ped[path_idx].dec_10_r[0], max_cons), bin2cons(ped[path_idx].dec_10_i[0], max_cons));

        ped[path_idx].dec_gray_r[0] = num2bit(bin2gray(ped[path_idx].dec_10_r[0]), qam_bit_per_axis);
        ped[path_idx].dec_gray_i[0] = num2bit(bin2gray(ped[path_idx].dec_10_i[0]), qam_bit_per_axis);
    }

    auto z = z_in;
    auto R = R_upper;
    // ped engine
    for (int i = 0; i < n_ss; ++i) {
        //std::cout << "layer: " << i << std::endl;

        if (i == 0) {
            for (int path_idx = 0; path_idx < path_num; ++path_idx) {
                std::complex<double> ped_tmp = z[i] - ic({R[0][0]}, {ped[path_idx].cons[0]}); 
                ped[path_idx].dist[i] = aabs(ped_tmp.real(), ped_tmp.imag()); 
                ped[path_idx].total_dist += ped[path_idx].dist[i];
            }
        } else {
            for (int path_idx = 0; path_idx < path_num; ++path_idx) {
                auto z_ic = z[i];
                for (int j = 0; j < i; ++j) {
                    z_ic -= ic({R[i][j]}, {ped[path_idx].cons[j]});
                }

                auto [ped_tmp, dec_cons] = slicer_c(R[i][i].real(), z_ic, slicer_layer);
                int x_idx_real = gray2bin(bit2num(dec_cons.first));
                int x_idx_imag = gray2bin(bit2num(dec_cons.second));

                ped[path_idx].dec_10_r[i] = x_idx_real;
                ped[path_idx].dec_10_i[i] = x_idx_imag;
                ped[path_idx].cons[i] = bin2cons(x_idx_real, max_cons) + std::complex<double>(0, 1) * bin2cons(x_idx_imag, max_cons);

                ped[path_idx].dec_gray_r[i] = dec_cons.first;
                ped[path_idx].dec_gray_i[i] = dec_cons.second;

                ped[path_idx].dist[i] = aabs(ped_tmp.first, ped_tmp.second);
                //ped[path_idx].total_dist += ped[path_idx].dist[i];
                ped[path_idx].total_dist = aabs(ped[path_idx].total_dist, ped[path_idx].dist[i]);
            }
        }
    }
    // 初始化 LLR 位元陣列
    double llr_max = 1000.0;
    std::vector<double> llr_bit_0(qam_bit * n_ss, llr_max);
    std::vector<double> llr_bit_1(qam_bit * n_ss, llr_max);
    
    for (int ss_idx = 0; ss_idx < n_ss; ++ss_idx) {
        int base_idx = ss_idx * qam_bit; // 提前計算重複使用的索引基礎
        for (int bit_idx = 0; bit_idx < qam_bit; ++bit_idx) {
            int llr_idx = base_idx + bit_idx; // 使用預計算的基礎來計算 llr_idx
            bool is_real_part = bit_idx < qam_bit_per_axis; // 提前判斷是處理實部還是虛部

            for (int path_idx = 0; path_idx < path_num; ++path_idx) {
                const auto& current_ped = ped[path_idx]; // 使用引用來簡化訪問
                int bit_value = is_real_part ? current_ped.dec_gray_r[ss_idx][bit_idx] : current_ped.dec_gray_i[ss_idx][bit_idx - qam_bit_per_axis];
                double& llr_bit_target = bit_value == 0 ? llr_bit_0[llr_idx] : llr_bit_1[llr_idx];

                // 更新 LLR 值
                if (current_ped.total_dist < llr_bit_target) {
                    llr_bit_target = current_ped.total_dist;
                }
            }
        }
    }

    // 計算 LLR 和 LLR 硬判斷 (Hard Decision)
    std::vector<double> llr(llr_bit_0.size());
    std::vector<int> llr_hd(llr_bit_0.size());

    std::transform(llr_bit_0.begin(), llr_bit_0.end(), llr_bit_1.begin(), llr.begin(), [](double a, double b) {
        return a - b;
    });

    std::transform(llr.begin(), llr.end(), llr_hd.begin(), [](double x) {
        return x >= 0 ? 1 : 0;
    });


    return {llr, llr_hd};


}


// demapper 函数
std::vector<double> demapper(double r, double y, int layers) {
    std::vector<double> llr(layers + 1, 0.0);
    llr[0] = r * y;
    double r_sqr = r * r;
    double threshold_base = (1 << layers) * r_sqr; // Use bit shift instead of pow
    for (int i = 0; i < layers; ++i) {
        llr[i + 1] = -std::abs(llr[i]) + threshold_base;
        threshold_base /= 2;
    }
    return llr;
}

// demapper_c 函数
std::vector<std::vector<double>> demapper_c(std::complex<double> r, std::complex<double> y, int layers) {
    std::vector<double> llr_r = demapper(std::real(r), std::real(y), layers);
    std::vector<double> llr_i = demapper(std::real(r), std::imag(y), layers);
    return {llr_r, llr_i};
}

std::vector<double> ml_dfe(const std::vector<std::vector<std::complex<double>>>& R, const std::vector<std::complex<double>>& z, int n_ss, int qam_bit, const std::vector<int>& ml_llr_hd, bool ml_dec_enb) {
    int qam_bit_per_axis = qam_bit / 2;
    int slicer_layer = qam_bit_per_axis - 1;
    int points_per_axis = 1 << qam_bit_per_axis; // Equivalent to 2^qam_bit_per_axis
    int max_cons = points_per_axis - 1;

    std::vector<double> llr_demapper;
    std::vector<std::complex<double>> llr_demapper_hd_cons(n_ss);

    for (int i = 0; i < n_ss; ++i) {
        //std::cout << "layer: " << i << std::endl;
        std::complex<double> z_ic = z[i];
        // Assume ic function adjusts z_ic accordingly
        for (int j = 0; j < i; ++j) {  
            z_ic -= ic({R[i][j]}, {llr_demapper_hd_cons[j]});
        }

        auto llr_demapper_tmp = demapper_c(R[i][i], z_ic, slicer_layer);
        
        std::vector<std::vector<int>> llr_demapper_hd_tmp;
        if (ml_dec_enb) {
            int idx_start = i * qam_bit;
            // Assuming ml_llr_hd contains hard decision bits
            for (int axis = 0; axis < 2; ++axis) {
                std::vector<int> temp;
                for (int bit = 0; bit < qam_bit_per_axis; ++bit) {
                    temp.push_back(ml_llr_hd[idx_start + axis * qam_bit_per_axis + bit]);
                }
                llr_demapper_hd_tmp.push_back(temp);
            }
        } else {
            for (const auto& sublist : llr_demapper_tmp) {
                std::vector<int> temp;
                for (double x : sublist) {
                    temp.push_back(x >= 0 ? 1 : 0);
                }
                llr_demapper_hd_tmp.push_back(temp);
            }
        }

        // Convert hard decisions into constellation indices
        //for (const auto& sublist : llr_demapper_hd_tmp) {
            int bin_real = bit2num(llr_demapper_hd_tmp[0]);
            int gray_real = gray2bin(bin_real);
            int bin_imag = bit2num(llr_demapper_hd_tmp[1]);
            int gray_imag = gray2bin(bin_imag);
            llr_demapper_hd_cons[i] = bin2cons(gray_real, max_cons) + std::complex<double>(0, 1) * bin2cons(gray_imag, max_cons);
        //}

        // Concatenate llr_demapper_tmp vectors to llr_demapper
        llr_demapper.insert(llr_demapper.end(), llr_demapper_tmp[0].begin(), llr_demapper_tmp[0].end());
        llr_demapper.insert(llr_demapper.end(), llr_demapper_tmp[1].begin(), llr_demapper_tmp[1].end());
    }

    return llr_demapper;
}

std::pair<std::vector<double>, std::vector<int>> mimo_srch(const std::vector<std::vector<std::complex<double>>>& R_upper, const std::vector<std::complex<double>>& z_in, int n_ss, int qam_bit, int path_num) {

        auto llr_fsd = run_hsinde(R_upper, z_in, n_ss, qam_bit, path_num);
        bool ml_dec_enb = true;
        auto llr_mldfe = ml_dfe(R_upper, z_in, n_ss, qam_bit, llr_fsd.second, ml_dec_enb);
        
        std::vector<double> llr_combined;  
        for (size_t i = 0; i < llr_fsd.first.size(); ++i) {
            // 取 llr_fsd 和 llr_dfe 的绝对值最小值
            double min_abs = std::min(std::abs(llr_fsd.first[i]), std::abs(llr_mldfe[i]));
            // 根据 llr_fsd 的符号决定正负
            double result = llr_fsd.first[i] >= 0 ? min_abs : -min_abs;
            llr_combined.push_back(result);
        }
        return {llr_combined, llr_fsd.second};

}
// 主函數
int main(int argc, char* argv[]) {
    // test of cand_gen
    int size = 8; // 代表序列的大小
    int path_num_t = 16; // 路徑數量
    int x_init_idx_real = 6; // 初始實部索引
    int x_init_idx_imag = 4; // 初始虛部索引

    // 使用 cand_gen 生成候選者列表
    auto cand = cand_gen(size, path_num_t, x_init_idx_real, x_init_idx_imag);

    // 打印生成的候選者列表
    for (const auto& pair : cand) {
        std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
    }


    // test of ic and aabs
    std::complex<double> a(1.0, 2.0); // 定义第一个复数
    std::complex<double> b(3.0, 4.0); // 定义第二个复数
    std::complex<double> result_ic = ic(a, b); // 计算两个复数的乘积
    std::cout << "Result: " << result_ic.real() << " + " << result_ic.imag() << "i" << " and its norm is ";
    std::cout << aabs(result_ic.real(), result_ic.imag()) << std::endl;

    // 定义一个包含多个测试复数的向量
    std::vector<std::complex<double>> test_inputs = {
        std::complex<double>(-0.1, 7.9),
        std::complex<double>(-1.1, 6.8),
        std::complex<double>(-2.1, 5.7),
        std::complex<double>(-3.1, 4.6),
        std::complex<double>(-4.1, 3.5),
        std::complex<double>(-5.1, 2.4),
        std::complex<double>(-6.1, 1.3),
        std::complex<double>(-7.1, 0.2),
    };

    int layers = 2; // 示例层数
    double r = 1.0;
    // 遍历所有测试输入
    for (const auto& y_c : test_inputs) {
        auto result = slicer_c(r, y_c, layers);
        auto& [peds, decisions] = result;
        auto& [ped_r, ped_i] = peds;
        auto& [dec_r, dec_i] = decisions;

        // 打印当前复数输入和对应的结果
        std::cout << "Input: " << y_c.real() << " + " << y_c.imag() << "i" << std::endl;
        std::cout << "PED Real: " << ped_r << ", PED Imag: " << ped_i << std::endl;
        std::cout << "Decisions Real: ";
        for (auto& d : dec_r) std::cout << d << " ";
        std::cout << "\nDecisions Imag: ";
        for (auto& d : dec_i) std::cout << d << " ";
        std::cout << "\n----------------------\n";
    }

    // test of bin2gray, gray2bin, bin2cons, num2bit, bit2num
    int bin_idx = 6;
    std::cout << "bin2gray of " << bin_idx << " is " << bin2gray(bin_idx) << std::endl;
    std::cout << "gray2bin of " << bin2gray(bin_idx) << " is " << gray2bin(bin2gray(bin_idx)) << std::endl;
    std::cout << "bin2cons of " << bin_idx << " is " << bin2cons(bin_idx, 7) << std::endl;
    auto bin_list = num2bit(bin_idx, 3);
    std::cout << "num2bit of " << bin_idx << " is [ ";
    for (auto& d: bin_list) std::cout << d <<" ";
    std:: cout << "]" << " ==> bit2num = " << bit2num(bin_list) << std::endl;


    std::vector<std::vector<std::complex<double>>> R_upper = {
        {std::complex<double>(0.2, 0), std::complex<double>(0, 0), std::complex<double>(0, 0), std::complex<double>(0, 0)},
        {std::complex<double>(1.1, -0.9), std::complex<double>(1.3, 0), std::complex<double>(0, 0), std::complex<double>(0, 0)},
        {std::complex<double>(0.7, 0.2), std::complex<double>(1.3, 0.9), std::complex<double>(0.6, 0), std::complex<double>(0, 0)},
        {std::complex<double>(-0.5, -0.4), std::complex<double>(-0.8, 0.3), std::complex<double>(0.3, 0.3), std::complex<double>(0.8, 0)}
    };

    std::vector<std::complex<double>> x_in = {
        std::complex<double>(-3.0,  1.0),
        std::complex<double>( -1.0, -1.0),
        std::complex<double>(-5.0,  1.0),
        std::complex<double>( 3.0,  7.0),
    };




    // 计算 z = R * x
    std::vector<std::complex<double>> z_in = dotProduct(R_upper, x_in);


    // 打印结果
    for (const auto& elem : z_in) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    // 设置其他参数
    int n_ss = std::stoi(argv[1]);       // 测试用的空间流数量
    int qam_bit = std::stoi(argv[2]);    // 64-QAM
    int path_num = pow(2, qam_bit); // 4096;  // 路径数量
    if (argc > 3){
        path_num = std::stoi(argv[3]); 
    }
    std::cout << "n_ss = " << n_ss << ", QAM = " << pow(2, qam_bit) << ", path_num = " << path_num << std::endl;
    
    // 调用 run_hsinde 函数
    //run_hsinde(R_upper, z_in, n_ss, qam_bit, path_num);


    std::vector<std::vector<std::complex<double>>> R_random(n_ss, std::vector<std::complex<double>>(n_ss));



    std::random_device rd;  // 隨機數生成器的種子
    std::mt19937 gen(rd()); // 標準 mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, 1); // 假設我們生成0到1之間的隨機數
    std::uniform_int_distribution<> distrib(0, 1); // 定義一個分佈在0和1之間的均勻分佈

    // 開始時間點
    auto start = std::chrono::high_resolution_clock::now();

    int n_trial = 15000;
    for (int n = 0; n < n_trial; n++){
        for (int i = 0; i < n_ss; ++i) {
            for (int j = 0; j <= i; ++j) {
                double realPart = dis(gen); // 生成實部
                double imagPart = dis(gen); // 生成虛部
                if (i == j) {
                    imagPart = 0.0;
                }
                R_random[i][j] = std::complex<double>(realPart, imagPart);
            }
        }
        
        std::vector<int> bit_random; 
        std::vector<std::complex<double>> x_in_random(n_ss); 
        for (int i = 0; i < n_ss; ++i) {
            for (int b_idx = 0; b_idx < qam_bit; b_idx++){
                bit_random.push_back(distrib(gen));
            }
            
        
            std::vector<std::vector<int>> bin_tmp;

            int idx_start = i * qam_bit;
            int qam_bit_per_axis = qam_bit/2;    
            int points_per_axis = 1 << qam_bit_per_axis; // Equivalent to 2^qam_bit_per_axis
            int max_cons = points_per_axis - 1;;
            
            for (int axis = 0; axis < 2; ++axis) {
                std::vector<int> temp;
                for (int bit = 0; bit < qam_bit_per_axis; ++bit) {
                    temp.push_back(bit_random[idx_start + axis * qam_bit_per_axis + bit]);
                }
                bin_tmp.push_back(temp);
            }
            int bin_real = bit2num(bin_tmp[0]);
            int gray_real = gray2bin(bin_real);
            int bin_imag = bit2num(bin_tmp[1]);
            int gray_imag = gray2bin(bin_imag);

            x_in_random[i] = bin2cons(gray_real, max_cons) + std::complex<double>(0, 1) * bin2cons(gray_imag, max_cons);
        }
                

        z_in = dotProduct(R_random, x_in_random);

        auto result = run_hsinde(R_random, z_in, n_ss, qam_bit, path_num);
        std::vector<double> llr = result.first;
        std::vector<int> llr_hd = result.second;
        
        bool ml_dec_enb = false;
        auto llr_mldfe = ml_dfe(R_random, z_in, n_ss, qam_bit, result.second, ml_dec_enb);  
        std::vector<int> llr_mldfe_hd(llr_mldfe.size()); 
        std::transform(llr_mldfe.begin(), llr_mldfe.end(), llr_mldfe_hd.begin(), [](double x) {
            return x >= 0 ? 1 : 0;
        });
        
        auto result_srch = mimo_srch(R_random, z_in, n_ss, qam_bit, path_num);
        std::vector<double> llr_srch = result_srch.first;
        std::vector<int> llr_srch_hd = result_srch.second;
        


        for (int i = 0; i < llr_hd.size(); i++){
            if (llr_hd[i] != bit_random[i]){
                std::cout << "\nFSD HD mismatch !!" << std::endl;
            }
            if (llr_mldfe_hd[i] != bit_random[i]){
                std::cout << "\nDFE HD mismatch !!" << std::endl;
            }
            if (llr_srch_hd[i] != bit_random[i]){
                std::cout << "\nSRCH HD mismatch !!" << std::endl;
            }
            
        }
        
    }
    
    std::cout << std::endl;
    // 結束時間點
    auto end = std::chrono::high_resolution_clock::now();

    // 計算並輸出執行時間
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "execution time: " << duration.count()*0.001 << " seconds" << std::endl;


    return 0;
}


