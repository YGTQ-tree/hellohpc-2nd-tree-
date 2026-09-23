T1
> B (LibreChat)

T2
> 4.3GB 3.4s

T3
> 838 419

T4
> 
0000:06:00.0 Signal processing controller [1180]: Huawei Technologies Co., Ltd. iBMA Virtual Network Adapter [19e5:1710] (rev 01)
0000:75:00.0 Network and computing encryption device [1000]: Huawei Technologies Co., Ltd. HiSilicon SEC Engine [19e5:a255] (rev 21)
0000:7d:00.0 Ethernet controller [0200]: Huawei Technologies Co., Ltd. HNS GE/10GE/25GE/50GE/100GE RDMA Network Controller [19e5:a226] (rev 21)
0000:b5:00.0 Network and computing encryption device [1000]: Huawei Technologies Co., Ltd. HiSilicon SEC Engine [19e5:a255] (rev 21)
0000:bd:00.0 Ethernet controller [0200]: Huawei Technologies Co., Ltd. HNS GE/10GE/25GE RDMA Network Controller [19e5:a222] (rev 21)
0000:bd:00.1 Ethernet controller [0200]: Huawei Technologies Co., Ltd. HNS GE/10GE/25GE RDMA Network Controller [19e5:a222] (rev 21)

===== enp125s0f0 =====
driver: hns3
version: 5.10.0-236.0.0.138.oe2203sp3.aa
firmware-version: 1.8.15.0
expansion-rom-version: 
bus-info: 0000:7d:00.0
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: yes
supports-priv-flags: yes
===== enp189s0f0 =====
driver: hns3
version: 5.10.0-236.0.0.138.oe2203sp3.aa
firmware-version: 1.8.15.0
expansion-rom-version: 
bus-info: 0000:bd:00.0
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: yes
supports-priv-flags: yes
===== enp189s0f1 =====
driver: hns3
version: 5.10.0-236.0.0.138.oe2203sp3.aa
firmware-version: 1.8.15.0
expansion-rom-version: 
bus-info: 0000:bd:00.1
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: yes
supports-priv-flags: yes
===== ibp1s0 =====
driver: mlx5_core[ib_ipoib]
version: 5.8-1.0.1
firmware-version: 20.35.1012 (MT_0000000237)
expansion-rom-version: 
bus-info: 0000:01:00.0
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
===== tun0 =====
driver: tun
version: 1.6
firmware-version: 
expansion-rom-version: 
bus-info: tun
supports-statistics: no
supports-test: no
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: no

___
bash-5.1$ echo "===== HOST ====="
===== HOST =====
bash-5.1$ hostname
kp001.pi.sjtu.edu.cn
bash-5.1$ 
bash-5.1$ echo "===== NIC ====="
===== NIC =====
bash-5.1$ for i in /sys/class/net/*; do
>     iface=$(basename "$i")
>     [ "$iface" = "lo" ] && continue
>     echo "----- $iface -----"
>     ethtool -i "$iface" 2>/dev/null
> done
----- enp125s0f0 -----
driver: hns3
version: 5.10.0-236.0.0.138.oe2203sp3.aa
firmware-version: 1.8.15.0
expansion-rom-version: 
bus-info: 0000:7d:00.0
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: yes
supports-priv-flags: yes
----- enp189s0f0 -----
driver: hns3
version: 5.10.0-236.0.0.138.oe2203sp3.aa
firmware-version: 1.8.15.0
expansion-rom-version: 
bus-info: 0000:bd:00.0
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: yes
supports-priv-flags: yes
----- enp189s0f1 -----
driver: hns3
version: 5.10.0-236.0.0.138.oe2203sp3.aa
firmware-version: 1.8.15.0
expansion-rom-version: 
bus-info: 0000:bd:00.1
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: yes
supports-priv-flags: yes
----- ibp1s0 -----
driver: mlx5_core[ib_ipoib]
version: 5.8-1.0.1
firmware-version: 20.35.1012 (MT_0000000237)
expansion-rom-version: 
bus-info: 0000:01:00.0
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
----- tun0 -----
driver: tun
version: 1.6
firmware-version: 
expansion-rom-version: 
bus-info: tun
supports-statistics: no
supports-test: no
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: no
bash-5.1$ 
bash-5.1$ echo "===== PCI ====="
===== PCI =====
bash-5.1$ lspci -Dnn | grep -Ei 'ethernet|network'
0000:06:00.0 Signal processing controller [1180]: Huawei Technologies Co., Ltd. iBMA Virtual Network Adapter [19e5:1710] (rev 01)
0000:75:00.0 Network and computing encryption device [1000]: Huawei Technologies Co., Ltd. HiSilicon SEC Engine [19e5:a255] (rev 21)
0000:7d:00.0 Ethernet controller [0200]: Huawei Technologies Co., Ltd. HNS GE/10GE/25GE/50GE/100GE RDMA Network Controller [19e5:a226] (rev 21)
0000:b5:00.0 Network and computing encryption device [1000]: Huawei Technologies Co., Ltd. HiSilicon SEC Engine [19e5:a255] (rev 21)
0000:bd:00.0 Ethernet controller [0200]: Huawei Technologies Co., Ltd. HNS GE/10GE/25GE RDMA Network Controller [19e5:a222] (rev 21)
0000:bd:00.1 Ethernet controller [0200]: Huawei Technologies Co., Ltd. HNS GE/10GE/25GE RDMA Network Controller [19e5:a222] (rev 21)

___

T5
gcc (GCC) 10.3.1

/usr/bin/gcc

ELF 头：
  Magic：  7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  类别:                              ELF64
  数据:                              2 补码，小端序 (little endian)
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI 版本:                          0
  类型:                              EXEC (可执行文件)
  系统架构:                          AArch64
  版本:                              0x1
  入口点地址：              0x400540
  程序头起点：              64 (bytes into file)
  Start of section headers:          68864 (bytes into file)
  标志：             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         9
  Size of section headers:           64 (bytes)
  Number of section headers:         30
  Section header string table index: 29

bash-5.1$ readelf -h "$(realpath "$(which gcc)")"
ELF 头：
  Magic：  7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  类别:                              ELF64
  数据:                              2 补码，小端序 (little endian)
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI 版本:                          0
  类型:                              EXEC (可执行文件)
  系统架构:                          AArch64
  版本:                              0x1
  入口点地址：              0x404440
  程序头起点：              64 (bytes into file)
  Start of section headers:          1119168 (bytes into file)
  标志：             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         9
  Size of section headers:           64 (bytes)
  Number of section headers:         28
  Section header string table index: 27


T7
> SHA256:f4e9f91cc7ffb0e4810744776728a8d863623bc518ab754ce90b6d1513720494