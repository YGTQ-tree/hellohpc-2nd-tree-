T1
> B (LibreChat)

T2
> 4.3GB 3.4s

T3
> 838.0 419.0

T4
> ~~enp189s0f1~~ ~~enp125s0f0~~enp189s0f0?

T5
> 10.3.1 0x404440
___
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


  bash-5.1$ echo "===== ADDR ====="
===== ADDR =====
bash-5.1$ ip -br addr
lo               UNKNOWN        127.0.0.1/8 ::1/128 
enp189s0f0       UP             172.16.32.1/16 fe80::12c3:abff:fed8:8ccc/64 
enp189s0f1       UP             202.120.58.251/28 2001:da8:8000:7020:2e92:b01d:ee72:d33a/64 fe80::a322:92e5:fcf2:6ac5/64 
enp125s0f0       DOWN           
ibp1s0           DOWN           10.0.32.1/16 
tun0             UNKNOWN        10.9.24.154/21 fe80::4ea7:aedc:53fe:6d17/64 
bash-5.1$ 
bash-5.1$ echo

bash-5.1$ echo "===== ROUTE ====="
===== ROUTE =====
bash-5.1$ ip route
default via 202.120.58.254 dev enp189s0f1 proto static metric 101 
10.0.0.0/16 dev ibp1s0 proto kernel scope link src 10.0.32.1 metric 150 linkdown 
10.9.24.0/21 dev tun0 proto kernel scope link src 10.9.24.154 
10.119.3.66 via 10.9.24.1 dev tun0 
10.119.9.12 via 10.9.24.1 dev tun0 
10.119.9.216 via 10.9.24.1 dev tun0 
10.119.10.236 via 10.9.24.1 dev tun0 
10.149.240.171 via 10.9.24.1 dev tun0 
111.186.38.22 via 10.9.24.1 dev tun0 
111.186.56.113 via 10.9.24.1 dev tun0 
111.186.56.229 via 10.9.24.1 dev tun0 
111.186.59.34 via 10.9.24.1 dev tun0 
172.16.0.0/16 dev enp189s0f0 proto kernel scope link src 172.16.32.1 metric 100 
192.168.0.0/16 via 172.16.0.1 dev enp189s0f0 proto static metric 100 
202.120.42.43 via 10.9.24.1 dev tun0 
202.120.46.51 via 10.9.24.1 dev tun0 
202.120.54.220 via 10.9.24.1 dev tun0 
202.120.58.240/28 dev enp189s0f1 proto kernel scope link src 202.120.58.251 metric 101 
202.121.181.130 via 10.9.24.1 dev tun0 
bash-5.1$ 
bash-5.1$ echo

bash-5.1$ echo "===== DEFAULT ====="
===== DEFAULT =====
bash-5.1$ ip route show default
default via 202.120.58.254 dev enp189s0f1 proto static metric 101 
bash-5.1$ 
bash-5.1$ echo

bash-5.1$ echo "===== ARMLOGIN IP ====="
===== ARMLOGIN IP =====
bash-5.1$ getent ahostsv4 armlogin.hpc.sjtu.edu.cn
202.120.58.251  STREAM armlogin.hpc.sjtu.edu.cn
202.120.58.251  DGRAM  
202.120.58.251  RAW    
bash-5.1$ 
bash-5.1$ echo

bash-5.1$ echo "===== PATH TO ARMLOGIN ====="
===== PATH TO ARMLOGIN =====
bash-5.1$ ARMIP=$(getent ahostsv4 armlogin.hpc.sjtu.edu.cn | awk 'NR==1{print $1}')
bash-5.1$ ip route get "$ARMIP"
local 202.120.58.251 dev lo src 202.120.58.251 uid 7254 
    cache <local> 

___

T6
> 38 ~~7537.5~~ 7539.2 GFLOPS

T7
> ~~SHA256:b261da718d74ff6f15cf5f5f3c42fbbe2166ab040a6838095547412c3dbfd584~~
>edaad70ee1670b56aeb090a5ebd038925f60146459b74924e0f8f5cc26eecca4
T8
> 50.0 37.5

T9
> 0.559 2.71