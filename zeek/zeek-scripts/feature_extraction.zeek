module FeatureExtraction;

export {
    redef record Conn::Info += {
        # Adding new fields as per UNSW-NB15 dataset
        srcip: addr &log &optional;
        sport: port &log &optional;
        dstip: addr &log &optional;
        dstport: port &log &optional;
        proto: string &log &optional;
        state: string &log &optional;
        dur: interval &log &optional;
        sbytes: count &log &optional;
        dbytes: count &log &optional;
        sttl: count &log &optional;
        dttl: count &log &optional;
        sloss: count &log &optional;
        dloss: count &log &optional;
        service: string &log &optional;
        Sload: double &log &optional;
        Dload: double &log &optional;
        Spkts: count &log &optional;
        Dpkts: count &log &optional;
        swin: count &log &optional;
        dwin: count &log &optional;
        stcpb: count &log &optional;
        dtcpb: count &log &optional;
        smeansz: double &log &optional;
        dmeansz: double &log &optional;
        trans_depth: count &log &optional;
        res_bdy_len: count &log &optional;
        Sjit: double &log &optional;
        Djit: double &log &optional;
        Stime: time &log &optional;
        Ltime: time &log &optional;
        Sintpkt: double &log &optional;
        Dintpkt: double &log &optional;
        tcprtt: double &log &optional;
        synack: double &log &optional;
        ackdat: double &log &optional;
        is_sm_ips_ports: bool &log &optional;
        ct_state_ttl: count &log &optional;
        ct_flw_http_mthd: count &log &optional;
        is_ftp_login: bool &log &optional;
        ct_ftp_cmd: count &log &optional;
        ct_srv_src: count &log &optional;
        ct_srv_dst: count &log &optional;
        ct_dst_ltm: count &log &optional;
        ct_src_ltm: count &log &optional;
        ct_src_dport_ltm: count &log &optional;
        ct_dst_sport_ltm: count &log &optional;
        ct_dst_src_ltm: count &log &optional;
        attack_cat: string &log &optional;
        Label: bool &log &optional;
    };
}

# Global tables for counts
global state_ttl_table: table[string] of count = table();
global flow_http_methods: table[string] of count = table();
global ftp_cmd_count: count = 0;
global srv_src_count: table[addr, string] of count = table();
global srv_dst_count: table[addr, string] of count = table();
global dst_ltm_count: table[addr] of count = table();
global src_ltm_count: table[addr] of count = table();
global src_dport_ltm_count: table[addr, port] of count = table();
global dst_sport_ltm_count: table[addr, port] of count = table();
global dst_src_ltm_count: table[addr, addr] of count = table();

event zeek_init()
    {
    # Configure logging
    Log::create_stream(Conn::LOG, [$columns=Conn::Info]);
    }

event new_connection(c: connection)
    {
    c$FeatureExtraction$Stime = network_time();
    }

event connection_state_remove(c: connection)
    {
    local rec = c$conn;

    # Basic connection features
    rec$srcip = c$id$orig_h;
    rec$sport = c$id$orig_p;
    rec$dstip = c$id$resp_h;
    rec$dstport = c$id$resp_p;
    rec$proto = c$id$resp_p == 80/tcp ? "http" : c$id$resp_p == 443/tcp ? "https" : string(c$proto);
    rec$state = c$history;
    rec$dur = c$duration;
    rec$sbytes = c$orig$bytes;
    rec$dbytes = c$resp$bytes;

    # TTL values (from SYN packets)
    rec$sttl = c$orig$ttl ? c$orig$ttl : 0;
    rec$dttl = c$resp$ttl ? c$resp$ttl : 0;

    # Packet loss (requires sequence number tracking, approximate)
    rec$sloss = c$orig$retrans ? c$orig$retrans : 0;
    rec$dloss = c$resp$retrans ? c$resp$retrans : 0;

    # Service identification
    rec$service = c$service ? c$service : "-";

    # Load calculations (bits per second)
    if ( c$duration > 0 sec )
        {
        rec$Sload = (c$orig$bytes * 8.0) / c$duration;
        rec$Dload = (c$resp$bytes * 8.0) / c$duration;
        }
    else
        {
        rec$Sload = 0.0;
        rec$Dload = 0.0;
        }

    # Packet counts
    rec$Spkts = c$orig$packets ? c$orig$packets : 0;
    rec$Dpkts = c$resp$packets ? c$resp$packets : 0;

    # TCP window sizes and sequence numbers (from SYN packets)
    rec$swin = c$orig$window ? c$orig$window : 0;
    rec$dwin = c$resp$window ? c$resp$window : 0;
    rec$stcpb = c$orig$seq ? c$orig$seq : 0;
    rec$dtcpb = c$resp$seq ? c$resp$seq : 0;

    # Mean packet sizes
    rec$smeansz = rec$Spkts > 0 ? rec$sbytes / rec$Spkts : 0;
    rec$dmeansz = rec$Dpkts > 0 ? rec$dbytes / rec$Dpkts : 0;

    # Transaction depth and response body length (from HTTP events)
    rec$trans_depth = c$http?$trans_depth ? c$http$trans_depth : 0;
    rec$res_bdy_len = c$http?$resp_body_len ? c$http$resp_body_len : 0;

    # Jitter calculations (requires packet timing)
    rec$Sjit = c$orig$jit ? c$orig$jit : 0.0;
    rec$Djit = c$resp$jit ? c$resp$jit : 0.0;

    # Start and last time
    rec$Stime = c$FeatureExtraction$Stime;
    rec$Ltime = network_time();

    # Inter-packet arrival times (average)
    rec$Sintpkt = rec$Spkts > 1 ? c$duration / (rec$Spkts - 1) : 0.0;
    rec$Dintpkt = rec$Dpkts > 1 ? c$duration / (rec$Dpkts - 1) : 0.0;

    # TCP round-trip times
    rec$tcprtt = c$rtt$duration ? c$rtt$duration : 0.0;
    rec$synack = c$rtt$synack ? c$rtt$synack : 0.0;
    rec$ackdat = c$rtt$ackdat ? c$rtt$ackdat : 0.0;

    # Same IPs and ports
    rec$is_sm_ips_ports = (c$id$orig_h == c$id$resp_h) && (c$id$orig_p == c$id$resp_p);

    # Counts over the last 100 connections (simplified example)
    local key = fmt("%s-%s", c$history, c$id$orig_h);
    state_ttl_table[key] += 1;
    rec$ct_state_ttl = state_ttl_table[key];

    # Attack category and label (requires external labeling)
    rec$attack_cat = "-";
    rec$Label = F;

    # Update counts for other features (simplified, needs proper implementation)
    srv_src_count[c$id$orig_h, rec$service] += 1;
    rec$ct_srv_src = srv_src_count[c$id$orig_h, rec$service];

    srv_dst_count[c$id$resp_h, rec$service] += 1;
    rec$ct_srv_dst = srv_dst_count[c$id$resp_h, rec$service];

    dst_ltm_count[c$id$resp_h] += 1;
    rec$ct_dst_ltm = dst_ltm_count[c$id$resp_h];

    src_ltm_count[c$id$orig_h] += 1;
    rec$ct_src_ltm = src_ltm_count[c$id$orig_h];

    src_dport_ltm_count[c$id$orig_h, c$id$resp_p] += 1;
    rec$ct_src_dport_ltm = src_dport_ltm_count[c$id$orig_h, c$id$resp_p];

    dst_sport_ltm_count[c$id$resp_h, c$id$orig_p] += 1;
    rec$ct_dst_sport_ltm = dst_sport_ltm_count[c$id$resp_h, c$id$orig_p];

    dst_src_ltm_count[c$id$orig_h, c$id$resp_h] += 1;
    rec$ct_dst_src_ltm = dst_src_ltm_count[c$id$orig_h, c$id$resp_h];

    # Write to Kafka
    Log::write(Conn::LOG, rec);
    }
