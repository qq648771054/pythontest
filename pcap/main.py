# -*- coding: utf-8 -*-

import dpkt
import socket
import datetime
from scapy.sendrecv import sniff
from scapy.utils import wrpcap

from winpcapy import WinPcapUtils

# Example Callback function to parse IP packets
def packet_callback(win_pcap, param, header, pkt_data):
    # Assuming IP (for real parsing use modules like dpkt)
    ip_frame = pkt_data[14:]
    # Parse ips
    src_ip = ".".join([str(ord(b)) for b in ip_frame[0xc:0x10]])
    dst_ip = ".".join([str(ord(b)) for b in ip_frame[0x10:0x14]])
    print "%d: %s -> %s" % (len(pkt_data), src_ip, dst_ip)

WinPcapUtils.capture_on("*Ethernet*", packet_callback)