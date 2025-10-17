# -*- coding: utf-8 -*-
# 文件: D:\python\visionhomework\ethernet_encap_gui.py
# 目标：修正为课本“变体”CRC-32，确保输入 hello world 输出 A4484EFF

import tkinter as tk
from tkinter import ttk, messagebox

class EthernetFrameEncapsulator:
    """
    GUI 以太网(802.3 变体)帧封装器
    关键差异（相对标准以太网）：
    - CRC 采用“变体”CRC-32：非反射 refin=False/refout=False，poly=0x04C11DB7，init=0x00000000，xorout=0x00000000
    - 参与 CRC 的字节：Dest(6)+Src(6)+Length(2)+Data(含填充到>=46B)，不含 Preamble(7)+SFD(1)
    - Length 字段 = 数据段实际发送长度(含填充) + 18（目的6+源6+长度2+FCS4），教材写法
    """

    def __init__(self):
        # === 创建窗口 ===
        self.root = tk.Tk()
        self.root.title("IEEE 802.3(教材变体) 以太网帧封装工具")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # === 样式 ===
        self._setup_styles()

        # === 输入控件 ===
        self._create_input()

        # === 操作按钮 ===
        self._create_actions()

        # === 输出区域 ===
        self._create_outputs()

    # -------------------- UI 相关 --------------------
    def _setup_styles(self):
        """设置界面样式（简洁黑体/等宽体）"""
        style = ttk.Style()
        style.configure('Title.TLabel',  font=('Arial', 14, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Normal.TLabel', font=('Arial', 10))
        style.configure('Mono.TLabel',   font=('Consolas', 10))

    def _create_input(self):
        """输入区域：目的/源MAC、数据、数据格式"""
        wrapper = ttk.Frame(self.root, padding=20)
        wrapper.grid(row=0, column=0, sticky='nsew')
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        wrapper.columnconfigure(0, weight=1)

        ttk.Label(wrapper, text="IEEE 802.3(教材变体) 以太网帧封装工具", style='Title.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 16))

        lf = ttk.LabelFrame(wrapper, text="输入参数", padding=12)
        lf.grid(row=1, column=0, sticky='ew')
        lf.columnconfigure(1, weight=1)

        # 目的 MAC（教材示例）
        ttk.Label(lf, text="目的MAC地址：", style='Header.TLabel').grid(row=0, column=0, sticky='w', pady=4)
        self.dest_mac_var = tk.StringVar(value="80:00:FF:60:2C:DC")
        ttk.Entry(lf, textvariable=self.dest_mac_var, font=('Consolas', 10)).grid(row=0, column=1, sticky='ew', padx=8)

        # 源 MAC（教材示例）
        ttk.Label(lf, text="源MAC地址：", style='Header.TLabel').grid(row=1, column=0, sticky='w', pady=4)
        self.src_mac_var = tk.StringVar(value="80:00:FE:85:3A:5F")
        ttk.Entry(lf, textvariable=self.src_mac_var, font=('Consolas', 10)).grid(row=1, column=1, sticky='ew', padx=8)

        # 数据（默认 hello world 以便你核对 A4484EFF）
        ttk.Label(lf, text="数据字段：", style='Header.TLabel').grid(row=2, column=0, sticky='w', pady=4)
        self.data_var = tk.StringVar(value="hello world")
        ttk.Entry(lf, textvariable=self.data_var, font=('Consolas', 10)).grid(row=2, column=1, sticky='ew', padx=8)

        # 格式说明
        ttk.Label(lf, text="说明：此工具按教材变体计算 CRC-32，非以太网标准 IEEE 版本。", style='Normal.TLabel').grid(row=3, column=0, columnspan=2, sticky='w', pady=(6, 0))

        self._container = wrapper  # 保存父容器供后续布局用

    def _create_actions(self):
        """操作按钮：封装、CRC自测、清空、退出"""
        bf = ttk.Frame(self._container)
        bf.grid(row=2, column=0, sticky='w', pady=12)

        ttk.Button(bf, text="封装帧",   command=self.encapsulate_frame).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(bf, text="CRC测试", command=self.run_crc_test).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(bf, text="清除输出", command=self.clear_output).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(bf, text="退出",   command=self.root.quit).grid(row=0, column=3)

    def _create_outputs(self):
        """输出区域：详细/十六进制/帧结构"""
        out = ttk.LabelFrame(self._container, text="输出结果", padding=12)
        out.grid(row=3, column=0, sticky='nsew')
        self._container.rowconfigure(3, weight=1)
        out.rowconfigure(0, weight=1)
        out.columnconfigure(0, weight=1)

        nb = ttk.Notebook(out)
        nb.grid(row=0, column=0, sticky='nsew')
        self.details_text = self._mk_tab(nb, "详细信息")
        self.hex_text     = self._mk_tab(nb, "十六进制")
        self.struct_text  = self._mk_tab(nb, "帧结构")

    def _mk_tab(self, nb, title):
        """创建一个只读文本选项卡"""
        f = ttk.Frame(nb)
        nb.add(f, text=title)
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        t = tk.Text(f, wrap='word', font=('Consolas', 10), state=tk.DISABLED)
        t.grid(row=0, column=0, sticky='nsew')
        sb = ttk.Scrollbar(f, orient=tk.VERTICAL, command=t.yview)
        sb.grid(row=0, column=1, sticky='ns')
        t.configure(yscrollcommand=sb.set)
        return t

    # -------------------- 辅助转换 --------------------
    def _mac_to_bytes(self, mac_str: str) -> bytes:
        """把 MAC 形如 XX:XX:XX:XX:XX:XX 转成 6 字节"""
        s = ''.join(c for c in mac_str if c.isalnum())
        if len(s) != 12:
            raise ValueError("MAC 地址必须是 12 个十六进制字符")
        return bytes.fromhex(s)

    # -------------------- CRC（核心修正） --------------------
    def _crc32_variant_nonref_init0(self, data: bytes) -> int:
        """
        教材“变体” CRC-32：
        - poly = 0x04C11DB7
        - refin = False, refout = False（非反射）
        - init = 0x00000000
        - xorout = 0x00000000
        - 以位序 MSB->LSB 左移实现
        返回：32 位无符号整数
        """
        poly = 0x04C11DB7
        crc  = 0x00000000  # 初值为 0
        for b in data:
            crc ^= (b << 24) & 0xFFFFFFFF
            for _ in range(8):
                if crc & 0x80000000:
                    crc = ((crc << 1) ^ poly) & 0xFFFFFFFF
                else:
                    crc = (crc << 1) & 0xFFFFFFFF
        return crc  # 无异或输出

    # -------------------- 主流程 --------------------
    def encapsulate_frame(self):
        """封装一帧（按教材流程），并给出各字段与 CRC-32(变体)"""
        try:
            dest = self._mac_to_bytes(self.dest_mac_var.get())
            src  = self._mac_to_bytes(self.src_mac_var.get())
            data = self.data_var.get().encode('utf-8')  # 本作业按字符串

            # 1) 数据段按规则补齐到 ≥46B
            if len(data) < 46:
                padded_data = data + b'\x00' * (46 - len(data))
            else:
                padded_data = data

            # 2) Length 按教材变体：数据(含填充) + 18
            length_value = len(padded_data) + 18
            length_bytes = length_value.to_bytes(2, 'big')

            # 3) 计算 CRC 输入（不含前导码和 SFD）
            crc_input = dest + src + length_bytes + padded_data

            # 4) 计算 CRC-32（变体），并转 4 字节大端
            crc32_val  = self._crc32_variant_nonref_init0(crc_input)
            crc32_be   = crc32_val.to_bytes(4, 'big')

            # 5) 构造完整物理层帧（仅用于展示；CRC 计算不含这两项）
            preamble = b'\xAA' * 7
            sfd      = b'\xAB'
            full_frame = preamble + sfd + crc_input + crc32_be

            # 6) 展示
            self._show_result(dest, src, length_bytes, data, padded_data, crc32_be, preamble, sfd, full_frame)

        except Exception as e:
            messagebox.showerror("错误", f"封装失败：{e}")

    # -------------------- 展示 --------------------
    def _show_result(self, dest, src, length, raw_data, padded_data, crc_be, preamble, sfd, full_frame):
        """把结果写到三个 Tab"""
        self._set_text(self.details_text, "")
        self._set_text(self.hex_text, "")
        self._set_text(self.struct_text, "")

        # 详细信息
        details = []
        details.append("=== 封装结果（教材变体 CRC-32）===")
        details.append(f"目的MAC：{self.dest_mac_var.get()}")
        details.append(f"源  MAC：{self.src_mac_var.get()}")
        details.append(f"原始数据长度：{len(raw_data)} B")
        details.append(f"填充后数据长度：{len(padded_data)} B")
        details.append(f"Length 字段(十六进制)：0x{length.hex().upper()}  (十进制 {int.from_bytes(length, 'big')})")
        details.append(f"CRC-32(变体) FCS：0x{crc_be.hex().upper()}  （4 字节大端显示）")
        details.append(f"帧总长度：{len(full_frame)} B  （含前导码7B + SFD1B + MAC头14B + 数据{len(padded_data)}B + FCS4B）")
        self._set_text(self.details_text, "\n".join(details))

        # 十六进制分解
        hx = []
        hx.append(f"完整帧({len(full_frame)}B) HEX：\n{self._hex(full_frame)}\n")
        hx.append("--- 字段分解 ---")
        hx.append(f"Preamble(7B)：{self._hex(preamble)}")
        hx.append(f"SFD(1B)：{sfd.hex().upper()}")
        hx.append(f"Destination(6B)：{self._hex(dest)}")
        hx.append(f"Source(6B)：{self._hex(src)}")
        hx.append(f"Length(2B)：{self._hex(length)}")
        hx.append(f"Data({len(padded_data)}B)：{self._hex(padded_data)}")
        hx.append(f"FCS(4B)：{self._hex(crc_be)}")
        self._set_text(self.hex_text, "\n".join(hx))

        # 结构图文字
        st = []
        st.append("帧结构（教材变体 802.3）：")
        st.append("+---------+---+-----------+-----------+--------+--------------------+----------+")
        st.append("|Preamble |SFD|Dest(6B)   |Src(6B)    |Len(2B) |Data(>=46B,含填充)  |FCS(4B)   |")
        st.append("+---------+---+-----------+-----------+--------+--------------------+----------+")
        st.append("注：CRC 输入为 Dest+Src+Length+Data，不含 Preamble 与 SFD。")
        st.append("    CRC-32 参数：poly=0x04C11DB7, refin=False, refout=False, init=0x00000000, xorout=0x00000000。")
        self._set_text(self.struct_text, "\n".join(st))

    # -------------------- CRC 自测（包含 “hello world”=A4484EFF） --------------------
    def run_crc_test(self):
        """用默认 MAC + 'hello world' 进行教辅校验，应得到 A4484EFF"""
        try:
            dest = self._mac_to_bytes(self.dest_mac_var.get())
            src  = self._mac_to_bytes(self.src_mac_var.get())
            data = b"hello world"
            padded = data + b'\x00' * (46 - len(data))
            length = (len(padded) + 18).to_bytes(2, 'big')
            crc_in = dest + src + length + padded
            crc32  = self._crc32_variant_nonref_init0(crc_in)
            msg = [
                "CRC-32(教材变体) 校验：",
                f"输入：Dest+Src+Length(0x{length.hex().upper()})+Data(含填充到46B)",
                f"结果：0x{crc32.to_bytes(4,'big').hex().upper()}",
                "期望：0xA4484EFF",
                "结论：{}".format("✓ 一致" if crc32.to_bytes(4,'big').hex().upper()=="A4484EFF" else "✗ 不一致")
            ]
            self._set_text(self.details_text, "\n".join(msg))
        except Exception as e:
            messagebox.showerror("测试错误", f"CRC 测试失败：{e}")

    # -------------------- 杂项 --------------------
    def clear_output(self):
        self._set_text(self.details_text, "")
        self._set_text(self.hex_text, "")
        self._set_text(self.struct_text, "")

    def _set_text(self, widget, content: str):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.config(state=tk.DISABLED)

    def _hex(self, b: bytes, group=2) -> str:
        s = b.hex().upper()
        return " ".join(s[i:i+group] for i in range(0, len(s), group))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    EthernetFrameEncapsulator().run()
