# defina_CRT_SCURE.NO-WARNING
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
    - 计算CRC-32校验和
    - 显示封装后的完整帧
    """

    def __init__(self):
        """
        初始化以太网帧封装器
        """
        self.root = tk.Tk()
        self.root.title("IEEE 802.3以太网帧封装工具 (最终修正版)")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        self.setup_styles()
        self.create_widgets()

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
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        main_frame.rowconfigure(4, weight=0)

        title_label = ttk.Label(main_frame, text="IEEE 802.3以太网帧封装工具",
                                style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 20), sticky=tk.W)

        self.create_input_section(main_frame)
        self.create_button_section(main_frame)
        self.create_output_section(main_frame)

        signature_label = ttk.Label(main_frame,
                                      text="作者：张钊洋 | 教学班：1216 | 学号：202313407492",
                                      style='Normal.TLabel',
                                      foreground='gray')
        signature_label.grid(row=4, column=0, pady=(20, 0), sticky="")

    def create_input_section(self, parent):
        """创建输入区域"""
        input_frame = ttk.LabelFrame(parent, text="输入参数", padding="15")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        input_frame.columnconfigure(1, weight=1)

        # 目的MAC地址
        ttk.Label(input_frame, text="目的MAC地址:",
                  style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dest_mac_var = tk.StringVar(value="80:00:FF:60:2C:DC")
        dest_mac_entry = ttk.Entry(input_frame, textvariable=self.dest_mac_var,
                                     font=('Consolas', 10))
        dest_mac_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))

        # 源MAC地址
        ttk.Label(input_frame, text="源MAC地址:",
                  style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.src_mac_var = tk.StringVar(value="80:00:FE:85:3A:5F")
        src_mac_entry = ttk.Entry(input_frame, textvariable=self.src_mac_var,
                                    font=('Consolas', 10))
        src_mac_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))

        # 数据字段
        ttk.Label(input_frame, text="数据字段:",
                  style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5)
        self.data_var = tk.StringVar(value="hello world")
        data_entry = ttk.Entry(input_frame, textvariable=self.data_var,
                                   font=('Consolas', 10))
        data_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))

    def create_button_section(self, parent):
        """创建操作按钮区域"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, pady=(0, 20), sticky=tk.W)

        self.encapsulate_btn = ttk.Button(button_frame, text="封装帧",
                                          command=self.encapsulate_frame)
        self.encapsulate_btn.grid(row=0, column=0, padx=(0, 10))
        clear_btn = ttk.Button(button_frame, text="清除输出",
                                 command=self.clear_output)
        clear_btn.grid(row=0, column=1, padx=(0, 10))
        exit_btn = ttk.Button(button_frame, text="退出",
                                command=self.root.quit)
        exit_btn.grid(row=0, column=2)

    def create_output_section(self, parent):
        """创建输出显示区域"""
        output_frame = ttk.LabelFrame(parent, text="输出结果", padding="15")
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(output_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.details_text = self.create_text_tab("详细信息")
        self.hex_text = self.create_text_tab("十六进制格式")
        self.structure_text = self.create_text_tab("帧结构")

    def create_text_tab(self, tab_name):
        """辅助函数，用于创建带滚动条的Text选项卡"""
        frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(frame, text=tab_name)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        text_widget = tk.Text(frame, wrap=tk.WORD, font=('Consolas', 10),
                              height=15, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        return text_widget

    def mac_to_bytes(self, mac_str):
        """将MAC地址字符串转换为字节"""
        try:
            mac_clean = ''.join(c for c in mac_str if c.isalnum())
            if len(mac_clean) != 12:
                raise ValueError("MAC地址必须包含12个十六进制字符")
            return bytes.fromhex(mac_clean)
        except Exception as e:
            raise ValueError(f"MAC地址格式无效: {str(e)}")

    def string_to_bytes(self, data_str):
        """将输入字符串根据所选格式转换为字节"""
        return data_str.encode('utf-8')

    # --- 最终修正的CRC函数 ---
    def calculate_crc32(self, data):
        """
        计算CRC-32校验和 (符合教材的非标准实现)
        非反射、初值=0x00000000、异或输出=0x00000000、生成多项式=0x04C11DB7
        """
        # 实现原始CRC-32计算，不进行位反转和最终异或
        POLYNOMIAL = 0x04C11DB7
        crc = 0x00000000  # 初值

        for byte in data:
            crc ^= (byte << 24)  # 将字节移到最高位
            for _ in range(8):    # 处理每一位
                if crc & 0x80000000:  # 检查最高位
                    crc = (crc << 1) ^ POLYNOMIAL
                else:
                    crc <<= 1
                crc &= 0xFFFFFFFF  # 确保32位

        return crc.to_bytes(4, byteorder='big')

    def encapsulate_frame(self):
        """执行帧封装的核心逻辑"""
        try:
            dest_mac = self.mac_to_bytes(self.dest_mac_var.get())
            src_mac = self.mac_to_bytes(self.src_mac_var.get())
            original_data = self.string_to_bytes(self.data_var.get())

            if len(original_data) > 1500:
                messagebox.showwarning("提示", "数据过长，此示例不支持分片。")
                return

            # 根据教材规则处理数据和长度字段
            data_with_padding = original_data
            length_value = len(original_data)

            if len(original_data) < 46:
                # 1. 如果数据长度小于46，则填充到46字节
                padding_len = 46 - len(original_data)
                data_with_padding = original_data + (b'\x00' * padding_len)
                
                # 2. 根据教材规则，此时长度字段应设为64
                length_value = 64
            
            # 将长度值转换为2字节
            length = length_value.to_bytes(2, byteorder='big')
            
            # 构建用于CRC计算的核心帧部分
            frame_core_for_crc = dest_mac + src_mac + length + data_with_padding
            
            # 计算CRC-32校验和
            crc = self.calculate_crc32(frame_core_for_crc)

            # 根据教材图片构建完整的物理层帧
            preamble = b'\xAA' * 7  # 前导码 (来自教材)
            sfd = b'\xAB'          # 帧起始定界符 (来自教材)
            full_frame = preamble + sfd + frame_core_for_crc + crc

            self.display_results(
                preamble=preamble, sfd=sfd, dest_mac=dest_mac, src_mac=src_mac,
                length=length, original_data=original_data, padded_data=data_with_padding,
                crc=crc, full_frame=full_frame
            )

        except Exception as e:
            messagebox.showerror("错误", f"封装失败: {str(e)}")

    def display_results(self, preamble, sfd, dest_mac, src_mac, length, original_data, padded_data, crc, full_frame):
        """显示封装结果"""
        for text_widget in [self.details_text, self.hex_text, self.structure_text]:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)

        # 详细信息
        details = (
            f"IEEE 802.3以太网帧封装结果\n{'=' * 60}\n\n"
            f"目的MAC地址: {self.format_mac(dest_mac)}\n"
            f"源MAC地址:   {self.format_mac(src_mac)}\n\n"
            f"数据字段 (原始长度):   {len(original_data)} 字节\n"
            f"数据字段 (填充后长度): {len(padded_data)} 字节\n"
            f"长度字段值 (十六进制): 0x{length.hex().upper()} ({int.from_bytes(length, 'big')})\n\n"
            f"CRC-32校验和 (FCS): 0x{crc.hex().upper()}\n\n"
            f"帧总长度: {len(full_frame)} 字节\n"
            f"  - 前导码 (Preamble):     7 字节\n"
            f"  - 帧起始定界符 (SFD):   1 字节\n"
            f"  - 目的/源 MAC:        6 + 6 字节\n"
            f"  - 长度:                 2 字节\n"
            f"  - 数据 (含填充):        {len(padded_data)} 字节\n"
            f"  - 帧校验序列 (FCS):     4 字节\n"
        )
        self.details_text.insert(tk.END, details)

        # 十六进制格式
        hex_output = (
            f"完整以太网帧 (十六进制格式, 共 {len(full_frame)} 字节):\n\n"
            f"{self.format_hex_with_spaces(full_frame.hex().upper())}\n\n{'=' * 60}\n\n分解显示:\n\n"
            f"前导码 (7B):      {self.format_hex_with_spaces(preamble.hex().upper())}\n"
            f"帧起始定界符 (1B):  {sfd.hex().upper()}\n"
            f"目的MAC (6B):     {self.format_hex_with_spaces(dest_mac.hex().upper())}\n"
            f"源MAC (6B):       {self.format_hex_with_spaces(src_mac.hex().upper())}\n"
            f"长度字段 (2B):      {self.format_hex_with_spaces(length.hex().upper())}\n"
            f"数据字段 ({len(padded_data)}B): {self.format_hex_with_spaces(padded_data.hex().upper())}\n"
            f"FCS校验和 (4B):   {self.format_hex_with_spaces(crc.hex().upper())}\n"
        )
        self.hex_text.insert(tk.END, hex_output)

        # 帧结构
        structure = (
            f"IEEE 802.3帧结构示意图:\n\n"
            f"+---------+---+------------------+------------------+--------+--------------------------+------------+\n"
            f"| Preamble|SFD| Destination MAC  |   Source MAC     | Length |           Data           |    FCS     |\n"
            f"|  (7 B)  |(1B)|      (6 B)       |      (6 B)       | (2 B)  |      (46-1500 B)       |   (4 B)    |\n"
            f"+---------+---+------------------+------------------+--------+--------------------------+------------+\n"
        )
        self.structure_text.insert(tk.END, structure)
        
        for text_widget in [self.details_text, self.hex_text, self.structure_text]:
            text_widget.config(state=tk.DISABLED)

    def format_hex_with_spaces(self, hex_str):
        """格式化十六进制字符串，添加空格"""
        return ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))

    def format_mac(self, mac_bytes):
        """将字节格式的MAC地址格式化"""
        return ':'.join(f'{b:02X}' for b in mac_bytes)

    def clear_output(self):
        """清除所有输出"""
        for text_widget in [self.details_text, self.hex_text, self.structure_text]:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.config(state=tk.DISABLED)

    def run(self):
        """启动主循环"""
        self.root.mainloop()

if __name__ == "__main__":
    app = EthernetFrameEncapsulator()
    app.run()