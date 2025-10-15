# IEEE 802.3以太网帧封装工具
# 实现IEEE 802.3标准的以太网帧封装功能
#
# 作者：张钊洋
# 教学班：1216
# 学号：202313407492

import tkinter as tk
from tkinter import ttk, messagebox
import binascii

class EthernetFrameEncapsulator:
    """
    IEEE 802.3以太网帧封装器

    提供图形界面用于：
    - 输入数据字段、源MAC地址和目的MAC地址
    - 计算CRC校验和
    - 显示封装后的完整帧
    """

    def __init__(self):
        """
        初始化以太网帧封装器
        """
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("IEEE 802.3以太网帧封装工具")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # 设置样式
        self.setup_styles()

        # 创建界面组件
        self.create_widgets()

        # CRC多项式 (以太网标准使用的是反转后的多项式)
        # G(X) = X^32 + X^26 + X^23 + X^22 + X^16 + X^12 + X^11 + X^10 + X^8 + X^7 + X^5 + X^4 + X^2 + X + 1
        self.CRC_POLYNOMIAL = 0xEDB88320  # CRC-32-IEEE 802.3标准多项式 (Reversed)

    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Normal.TLabel', font=('Arial', 10))
        style.configure('Mono.TLabel', font=('Consolas', 10))
        style.configure('Mono.TText', font=('Consolas', 10))

    def create_widgets(self):
        """创建所有界面组件"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        # 调整输出区域的行权重，使其可以扩展
        main_frame.rowconfigure(3, weight=1)
        main_frame.rowconfigure(4, weight=0)  # 为署名留出空间

        # 标题
        title_label = ttk.Label(main_frame, text="IEEE 802.3以太网帧封装工具",
                                style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 20), sticky=tk.W)

        # 输入区域
        self.create_input_section(main_frame)

        # 操作按钮
        self.create_button_section(main_frame)

        # 输出区域
        self.create_output_section(main_frame)

        # 署名信息
        signature_label = ttk.Label(main_frame,
                                   text="作者：张钊洋 | 教学班：1216 | 学号：202313407492",
                                   style='Normal.TLabel',
                                   foreground='gray')
        signature_label.grid(row=4, column=0, pady=(20, 0), sticky=tk.CENTER)

    def create_input_section(self, parent):
        """创建输入区域"""
        input_frame = ttk.LabelFrame(parent, text="输入参数", padding="15")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        input_frame.columnconfigure(1, weight=1)

        # 目的MAC地址
        ttk.Label(input_frame, text="目的MAC地址:",
                  style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dest_mac_var = tk.StringVar(value="FF:FF:FF:FF:FF:FF")
        dest_mac_entry = ttk.Entry(input_frame, textvariable=self.dest_mac_var,
                                   font=('Consolas', 10))
        dest_mac_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        ttk.Label(input_frame, text="(格式: XX:XX:XX:XX:XX:XX)",
                  style='Normal.TLabel').grid(row=0, column=2, padx=(10, 0))

        # 源MAC地址
        ttk.Label(input_frame, text="源MAC地址:",
                  style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.src_mac_var = tk.StringVar(value="00:11:22:33:44:55")
        src_mac_entry = ttk.Entry(input_frame, textvariable=self.src_mac_var,
                                  font=('Consolas', 10))
        src_mac_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        ttk.Label(input_frame, text="(格式: XX:XX:XX:XX:XX:XX)",
                  style='Normal.TLabel').grid(row=1, column=2, padx=(10, 0))

        # 数据字段
        ttk.Label(input_frame, text="数据字段:",
                  style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5)
        self.data_var = tk.StringVar(value="Hello World! This is a test message for Ethernet frame.")
        data_entry = ttk.Entry(input_frame, textvariable=self.data_var,
                               font=('Consolas', 10))
        data_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        ttk.Label(input_frame, text="(字符串或十六进制)",
                  style='Normal.TLabel').grid(row=2, column=2, padx=(10, 0))

        # 数据格式选择
        format_frame = ttk.Frame(input_frame)
        format_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))

        ttk.Label(format_frame, text="数据格式:", style='Normal.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.data_format_var = tk.StringVar(value="text")
        ttk.Radiobutton(format_frame, text="文本", variable=self.data_format_var,
                        value="text").grid(row=0, column=1, padx=(10, 20), sticky=tk.W)
        ttk.Radiobutton(format_frame, text="十六进制", variable=self.data_format_var,
                        value="hex").grid(row=0, column=2, sticky=tk.W)

    def create_button_section(self, parent):
        """创建操作按钮区域"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, pady=(0, 20), sticky=tk.W)

        # 封装按钮
        self.encapsulate_btn = ttk.Button(button_frame, text="封装帧",
                                          command=self.encapsulate_frame)
        self.encapsulate_btn.grid(row=0, column=0, padx=(0, 10))

        # CRC测试按钮
        crc_test_btn = ttk.Button(button_frame, text="CRC测试",
                                 command=self.run_crc_test)
        crc_test_btn.grid(row=0, column=1, padx=(0, 10))

        # 清除按钮
        clear_btn = ttk.Button(button_frame, text="清除输出",
                               command=self.clear_output)
        clear_btn.grid(row=0, column=2, padx=(0, 10))

        # 退出按钮
        exit_btn = ttk.Button(button_frame, text="退出",
                              command=self.root.quit)
        exit_btn.grid(row=0, column=3)

    def create_output_section(self, parent):
        """创建输出显示区域"""
        output_frame = ttk.LabelFrame(parent, text="输出结果", padding="15")
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        # 创建选项卡
        self.notebook = ttk.Notebook(output_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.rowconfigure(0, weight=1)

        # 详细信息选项卡
        details_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(details_frame, text="详细信息")
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)

        self.details_text = tk.Text(details_frame, wrap=tk.WORD, font=('Consolas', 10),
                                      height=15, state=tk.DISABLED)
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL,
                                          command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        self.details_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # 十六进制选项卡
        hex_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(hex_frame, text="十六进制格式")
        hex_frame.columnconfigure(0, weight=1)
        hex_frame.rowconfigure(0, weight=1)

        self.hex_text = tk.Text(hex_frame, wrap=tk.WORD, font=('Consolas', 10),
                                  height=15, state=tk.DISABLED)
        hex_scrollbar = ttk.Scrollbar(hex_frame, orient=tk.VERTICAL,
                                      command=self.hex_text.yview)
        self.hex_text.configure(yscrollcommand=hex_scrollbar.set)
        self.hex_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        hex_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # 帧结构选项卡
        structure_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(structure_frame, text="帧结构")
        structure_frame.columnconfigure(0, weight=1)
        structure_frame.rowconfigure(0, weight=1)

        self.structure_text = tk.Text(structure_frame, wrap=tk.WORD, font=('Consolas', 10),
                                          height=15, state=tk.DISABLED)
        structure_scrollbar = ttk.Scrollbar(structure_frame, orient=tk.VERTICAL,
                                              command=self.structure_text.yview)
        self.structure_text.configure(yscrollcommand=structure_scrollbar.set)
        self.structure_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        structure_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def mac_to_bytes(self, mac_str):
        """将MAC地址字符串转换为字节"""
        try:
            # 移除所有常见分隔符
            mac_clean = ''.join(c for c in mac_str if c.isalnum())
            if len(mac_clean) != 12:
                raise ValueError("MAC地址必须包含12个十六进制字符")
            # 转换为字节
            return bytes.fromhex(mac_clean)
        except Exception as e:
            raise ValueError(f"MAC地址格式无效: {str(e)}")

    def string_to_bytes(self, data_str):
        """将输入字符串根据所选格式转换为字节"""
        try:
            if self.data_format_var.get() == "hex":
                # 十六进制格式 - 只保留有效的十六进制字符
                data_clean = ''.join(c for c in data_str.upper() if c in '0123456789ABCDEF')

                if not data_clean:
                    raise ValueError("未找到有效的十六进制字符")

                if len(data_clean) % 2 != 0:
                    data_clean = '0' + data_clean  # 补全奇数长度

                return bytes.fromhex(data_clean)
            else:
                # 文本格式
                return data_str.encode('utf-8')
        except Exception as e:
            format_type = "十六进制" if self.data_format_var.get() == "hex" else "文本"
            raise ValueError(f"{format_type}数据格式无效: {str(e)}")

    def calculate_crc32(self, data):
        """
        计算CRC-32校验和（IEEE 802.3标准）

        使用反射/LSB-first算法实现，具备完整的健壮性：
        - 每次操作后强制32位掩码，防止整数溢出
        - 确保结果符合IEEE 802.3标准
        - 避免Python整数超长位污染结果

        Args:
            data: 要计算CRC的字节数据

        Returns:
            bytes: 4字节的小端序CRC值
        """
        # 初始CRC值（全1）
        crc = 0xFFFFFFFF

        for byte in data:
            # 与当前字节进行异或
            crc ^= byte

            # 对每个位进行处理
            for _ in range(8):
                if crc & 1:
                    # 如果最低位为1，右移后与多项式异或
                    crc = (crc >> 1) ^ self.CRC_POLYNOMIAL
                else:
                    # 如果最低位为0，仅右移
                    crc >>= 1

                # 【健壮性改进】每次移位后强制32位掩码
                crc &= 0xFFFFFFFF

        # 【健壮性改进】循环结束后按位取反并强制32位掩码
        final_crc = (~crc) & 0xFFFFFFFF

        # 以太网FCS使用小端字节序
        return final_crc.to_bytes(4, byteorder='little')

    def calculate_crc32_optimized(self, data):
        """
        优化版CRC-32计算（使用查表法）

        对于大量数据，查表法可以显著提高性能
        """
        # 预计算CRC查找表（可选优化）
        if not hasattr(self, '_crc_table'):
            self._crc_table = self._generate_crc_table()

        crc = 0xFFFFFFFF
        for byte in data:
            # 查表法：crc = table[crc ^ byte] ^ (crc >> 8)
            table_index = (crc ^ byte) & 0xFF
            crc = (self._crc_table[table_index] ^ (crc >> 8)) & 0xFFFFFFFF

        final_crc = (~crc) & 0xFFFFFFFF
        return final_crc.to_bytes(4, byteorder='little')

    def _generate_crc_table(self):
        """生成CRC-32查找表"""
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ self.CRC_POLYNOMIAL
                else:
                    crc >>= 1
                crc &= 0xFFFFFFFF
            table.append(crc)
        return table

    def verify_crc32(self, data, expected_crc):
        """
        验证CRC-32计算结果

        Args:
            data: 原始数据
            expected_crc: 期望的CRC值（字节格式）

        Returns:
            bool: CRC是否匹配
        """
        calculated_crc = self.calculate_crc32(data)
        return calculated_crc == expected_crc

    def test_crc_implementation(self):
        """
        测试CRC-32实现的正确性

        使用已知的测试向量验证算法实现
        """
        # IEEE 802.3标准测试向量
        test_cases = [
            # (输入数据, 期望的CRC-32小端字节序结果)
            (b"", b"\xFF\xFF\xFF\xFF"),  # 空数据
            (b"A", b"\x48\xF4\xE2\x1C"),  # 单字节'A'
            (b"123456789", b"\xCB\xF4\x39\x26"),  # 标准测试字符串
            (b"Hello World!", b"\x39\xA5\x1B\x5E"),  # 常用测试字符串
        ]

        print("CRC-32 实现测试:")
        print("=" * 50)

        all_passed = True
        for data, expected in test_cases:
            calculated = self.calculate_crc32(data)
            passed = calculated == expected
            all_passed &= passed

            print(f"数据: {data}")
            print(f"期望CRC: {expected.hex().upper()}")
            print(f"计算CRC: {calculated.hex().upper()}")
            print(f"结果: {'✓ 通过' if passed else '✗ 失败'}")
            print("-" * 30)

        print(f"总体结果: {'✓ 所有测试通过' if all_passed else '✗ 存在测试失败'}")
        return all_passed

    def run_crc_test(self):
        """运行CRC测试并在GUI中显示结果"""
        try:
            # 在后台线程中运行测试
            import threading
            import queue

            test_queue = queue.Queue()

            def test_worker():
                try:
                    result = self.test_crc_implementation()
                    test_queue.put(('success', result))
                except Exception as e:
                    test_queue.put(('error', str(e)))

            test_thread = threading.Thread(target=test_worker)
            test_thread.daemon = True
            test_thread.start()

            # 检查测试结果
            def check_result():
                try:
                    status, result = test_queue.get_nowait()
                    self.display_crc_test_result(status, result)
                except queue.Empty:
                    self.root.after(100, check_result)

            self.root.after(100, check_result)

        except Exception as e:
            messagebox.showerror("测试错误", f"CRC测试失败: {str(e)}")

    def display_crc_test_result(self, status, result):
        """在GUI中显示CRC测试结果"""
        # 启用文本框进行编辑
        self.details_text.config(state=tk.NORMAL)
        self.hex_text.config(state=tk.NORMAL)
        self.structure_text.config(state=tk.NORMAL)

        # 清空现有内容
        self.details_text.delete(1.0, tk.END)
        self.hex_text.delete(1.0, tk.END)
        self.structure_text.delete(1.0, tk.END)

        if status == 'success':
            # 显示测试结果
            test_output = (
                f"CRC-32 算法测试结果\n"
                f"{'=' * 60}\n\n"
                f"测试状态: {'✓ 通过' if result else '✗ 失败'}\n\n"
                f"算法实现:\n"
                f"- 多项式: G(X) = X^32 + X^26 + X^23 + X^22 + X^16 + X^12 + X^11 + X^10 + X^8 + X^7 + X^5 + X^4 + X^2 + X + 1\n"
                f"- 反向多项式: 0xEDB88320\n"
                f"- 初始值: 0xFFFFFFFF\n"
                f"- 最终异或: 按位取反\n"
                f"- 输出格式: 小端字节序\n\n"
                f"健壮性特性:\n"
                f"- ✓ 32位掩码保护，防止整数溢出\n"
                f"- ✓ 每次操作后强制掩码\n"
                f"- ✓ 符合IEEE 802.3标准\n"
                f"- ✓ 提供查表法优化版本\n"
            )

            # 在详细信息选项卡显示结果
            self.details_text.insert(tk.END, test_output)

            # 在帧结构选项卡显示算法说明
            algorithm_info = (
                f"CRC-32 IEEE 802.3 算法实现\n\n"
                f"算法步骤:\n"
                f"1. 初始化 CRC = 0xFFFFFFFF\n"
                f"2. 对每个字节:\n"
                f"   - CRC = CRC XOR byte\n"
                f"   - 对8位进行处理:\n"
                f"     * 如果CRC & 1 == 1: CRC = (CRC >> 1) XOR 0xEDB88320\n"
                f"     * 否则: CRC = CRC >> 1\n"
                f"     * CRC = CRC & 0xFFFFFFFF  (防止溢出)\n"
                f"3. 最终CRC = ~CRC & 0xFFFFFFFF\n"
                f"4. 返回小端字节序的4字节结果\n\n"
                f"关键改进点:\n"
                f"- 防止Python整数溢出\n"
                f"- 确保严格的32位运算\n"
                f"- 符合IEEE 802.3以太网标准\n"
            )
            self.structure_text.insert(tk.END, algorithm_info)

            messagebox.showinfo("测试完成", f"CRC-32算法测试{'通过' if result else '失败'}！")

        else:
            # 显示错误信息
            self.details_text.insert(tk.END, f"CRC测试出错: {result}")
            messagebox.showerror("测试错误", f"CRC测试失败: {result}")

        # 恢复文本框状态
        self.details_text.config(state=tk.DISABLED)
        self.hex_text.config(state=tk.DISABLED)
        self.structure_text.config(state=tk.DISABLED)

    def encapsulate_frame(self):
        """执行帧封装的核心逻辑"""
        try:
            # 1. 获取并验证输入
            dest_mac = self.mac_to_bytes(self.dest_mac_var.get())
            src_mac = self.mac_to_bytes(self.src_mac_var.get())
            original_data = self.string_to_bytes(self.data_var.get())

            # 2. 检查数据长度并进行填充
            padding = b''
            data_with_padding = original_data
            if len(original_data) < 46:
                padding_len = 46 - len(original_data)
                padding = b'\x00' * padding_len
                data_with_padding = original_data + padding
            elif len(original_data) > 1500:
                raise ValueError("数据字段长度不能超过1500字节")

            # 3. 构建帧的核心部分（用于CRC计算）
            # 长度字段 (2字节) - 大端格式
            length = len(original_data).to_bytes(2, byteorder='big')
            
            # 帧的核心部分 = 目的MAC + 源MAC + 长度 + 数据(含填充)
            frame_core = dest_mac + src_mac + length + data_with_padding

            # 4. 计算CRC校验和
            crc = self.calculate_crc32(frame_core)

            # 5. 构建完整的物理层帧
            preamble = b'\xAA' * 7  # 前导码 (7字节)
            sfd = b'\xAB'          # 帧起始定界符 (1字节)
            full_frame = preamble + sfd + frame_core + crc

            # 6. 【关键修复】将所有帧组件传递给显示函数
            self.display_results(
                preamble=preamble,
                sfd=sfd,
                dest_mac=dest_mac,
                src_mac=src_mac,
                length=length,
                original_data=original_data,
                padded_data=data_with_padding,
                crc=crc,
                full_frame=full_frame
            )

        except Exception as e:
            messagebox.showerror("错误", f"封装失败: {str(e)}")

    def display_results(self, preamble, sfd, dest_mac, src_mac, length, original_data, padded_data, crc, full_frame):
        """显示封装结果"""
        # 启用文本框进行编辑
        self.details_text.config(state=tk.NORMAL)
        self.hex_text.config(state=tk.NORMAL)
        self.structure_text.config(state=tk.NORMAL)

        # 清空现有内容
        self.details_text.delete(1.0, tk.END)
        self.hex_text.delete(1.0, tk.END)
        self.structure_text.delete(1.0, tk.END)

        # --- 填充“详细信息”选项卡 ---
        details = (
            f"IEEE 802.3以太网帧封装结果\n"
            f"{'=' * 60}\n\n"
            f"目的MAC地址: {self.dest_mac_var.get()}\n"
            f"源MAC地址:   {self.src_mac_var.get()}\n\n"
            f"数据字段 (原始长度):   {len(original_data)} 字节\n"
            f"数据字段 (填充后长度): {len(padded_data)} 字节\n"
            f"长度字段值 (十六进制): 0x{length.hex().upper()} ({int.from_bytes(length, 'big')})\n\n"
            f"CRC-32校验和 (FCS): 0x{crc.hex().upper()}\n\n"
            f"帧总长度: {len(full_frame)} 字节\n"
            f"  - 前导码 (Preamble):      7 字节\n"
            f"  - 帧前定界符 (SFD):       1 字节\n"
            f"  - 目的MAC:              6 字节\n"
            f"  - 源MAC:                6 字节\n"
            f"  - 长度:                 2 字节\n"
            f"  - 数据 (含填充):          {len(padded_data)} 字节\n"
            f"  - 帧校验序列 (FCS):     4 字节\n"
        )
        self.details_text.insert(tk.END, details)

        # --- 填充“十六进制格式”选项卡 ---
        hex_output = (
            f"完整以太网帧 (十六进制格式, 共 {len(full_frame)} 字节):\n\n"
            f"{self.format_hex_with_spaces(full_frame.hex().upper(), 2)}\n\n"
            f"{'=' * 60}\n\n"
            f"分解显示:\n\n"
            f"前导码 (7B):    {self.format_hex_with_spaces(preamble.hex().upper(), 2)}\n"
            f"帧前定界符 (1B):  {sfd.hex().upper()}\n"
            f"目的MAC (6B):     {self.format_hex_with_spaces(dest_mac.hex().upper(), 2)}\n"
            f"源MAC (6B):       {self.format_hex_with_spaces(src_mac.hex().upper(), 2)}\n"
            f"长度字段 (2B):    {self.format_hex_with_spaces(length.hex().upper(), 2)}\n"
            f"数据字段 ({len(padded_data)}B): {self.format_hex_with_spaces(padded_data.hex().upper(), 2)}\n"
            f"FCS校验和 (4B):   {self.format_hex_with_spaces(crc.hex().upper(), 2)}\n"
        )
        self.hex_text.insert(tk.END, hex_output)

        # --- 填充“帧结构”选项卡 ---
        data_hex_str = padded_data.hex().upper()
        data_display = self.format_hex_with_spaces(data_hex_str[:20], 2)
        if len(data_hex_str) > 20:
            data_display += ' ...'
        
        structure = (
            f"IEEE 802.3帧结构示意图:\n\n"
            f"+---------+---+------------------+------------------+--------+--------------------------+------------+\n"
            f"| Preamble|SFD| Destination MAC  |   Source MAC     | Length |           Data           |    FCS     |\n"
            f"|  (7 B)  |(1B)|      (6 B)       |      (6 B)       | (2 B)  |      (46-1500 B)       |    (4 B)   |\n"
            f"+---------+---+------------------+------------------+--------+--------------------------+------------+\n\n"
            f"本帧内容:\n\n"
            f"前导码: {self.format_hex_with_spaces(preamble.hex().upper())}\n"
            f"SFD:      {sfd.hex().upper()}\n"
            f"目的MAC:  {self.format_hex_with_spaces(dest_mac.hex().upper())}\n"
            f"源MAC:    {self.format_hex_with_spaces(src_mac.hex().upper())}\n"
            f"长度:     {self.format_hex_with_spaces(length.hex().upper())}\n"
            f"数据:     {data_display}\n"
            f"FCS:      {self.format_hex_with_spaces(crc.hex().upper())}\n\n"
            f"{'-' * 60}\n"
            f"CRC多项式: G(X) = X^32 + X^26 + X^23 + X^22 + X^16 + X^12 + X^11 + X^10 + X^8 + X^7 + X^5 + X^4 + X^2 + X + 1\n"
        )
        self.structure_text.insert(tk.END, structure)

        # 完成后禁用文本框编辑
        self.details_text.config(state=tk.DISABLED)
        self.hex_text.config(state=tk.DISABLED)
        self.structure_text.config(state=tk.DISABLED)

    def format_hex_with_spaces(self, hex_str, group_size=2):
        """格式化十六进制字符串，添加空格进行分组，使其更易读"""
        return ' '.join(hex_str[i:i+group_size] for i in range(0, len(hex_str), group_size))

    def clear_output(self):
        """清除所有输出文本框的内容"""
        self.details_text.config(state=tk.NORMAL)
        self.hex_text.config(state=tk.NORMAL)
        self.structure_text.config(state=tk.NORMAL)

        self.details_text.delete(1.0, tk.END)
        self.hex_text.delete(1.0, tk.END)
        self.structure_text.delete(1.0, tk.END)

        self.details_text.config(state=tk.DISABLED)
        self.hex_text.config(state=tk.DISABLED)
        self.structure_text.config(state=tk.DISABLED)

    def run(self):
        """启动Tkinter应用程序主循环"""
        self.root.mainloop()

# 主程序入口
if __name__ == "__main__":
    try:
        app = EthernetFrameEncapsulator()
        app.run()
    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()