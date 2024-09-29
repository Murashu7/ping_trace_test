import pytest
from unittest.mock import patch, AsyncMock
import os
import subprocess
import pandas as pd
from unittest import mock
import asyncio
import shutil

# テスト対象の関数が記載されているモジュールをインポートします
from script import save_results, extract_ips, check_route_match, validate_ping, validate_route, ping, ping_multiple_hosts, \
    traceroute, trace_multiple_hosts, convert_to_list, select_kyoten, select_test_type, get_test_type_path, SelectionError, \
    generate_unique_filename, get_name_by_host, evaluate_ping_result, validate_ping_windows, is_windows, is_mac, UnsupportedOSError, \
    UnsupportedLanguageError, validate_ping_mac, check_os_and_language, is_japanese_os, is_english_os, ping_and_validate, process_ping, \
    trace_and_validate, process_trace, ping_trace_multiple_hosts, create_unique_folder
    
@pytest.mark.asyncio
async def test_save_results_ping(tmpdir):
    # 一時ディレクトリを作成
    results_dir = tmpdir.mkdir("results")

    # テストデータ
    kyoten_name = "拠点A"
    test_type = "テスト"
    index = 1
    host = "8.8.8.8"
    ping_output = "5 packets transmitted, 5 packets received, 0.0% packet loss"
    result_type = "ping"

    # save_results関数を実行
    await save_results(kyoten_name, test_type, index, host, ping_output, str(results_dir), result_type)

    # 期待されるファイルパス
    expected_file_path = os.path.join(str(results_dir), f'{index}_{kyoten_name}_{test_type}の結果.log')

    # ファイルが作成されたか確認
    assert os.path.exists(expected_file_path)

    # ファイルの内容を確認
    with open(expected_file_path, 'r') as f:
        content = f.read()
        assert f"{host} への {result_type} 結果" in content
        assert ping_output in content
        assert "="*40 in content
  
@pytest.mark.asyncio
async def test_save_results_traceroute(tmpdir):
    # 一時ディレクトリを作成
    results_dir = tmpdir.mkdir("results")

    # テストデータ
    kyoten_name = "拠点A"
    test_type = "テスト"
    index = 1
    host = "8.8.8.8"
    trace_output = "traceroute to 8.8.8.8, 30 hops max"
    result_type = "traceroute"

    # save_results関数を実行
    await save_results(kyoten_name, test_type, index, host, trace_output, str(results_dir), result_type)

    # 期待されるファイルパス
    expected_file_path = os.path.join(str(results_dir), f'{index}_{kyoten_name}_{test_type}の結果.log')

    # ファイルが作成されたか確認
    assert os.path.exists(expected_file_path)

    # ファイルの内容を確認
    with open(expected_file_path, 'r') as f:
        content = f.read()
        assert f"{host} への {result_type} 結果" in content
        assert trace_output in content
        assert "="*40 in content

class TestPing:

    @pytest.mark.asyncio
    async def test_ping_success(self):
        """pingコマンドが成功し、標準出力を返す場合のテスト"""
        cmd_args = ['ping', '-c', '1', '8.8.8.8']

        # 非同期プロセスをモックする
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_proc:
            mock_proc.return_value.communicate = AsyncMock(return_value=(b'PING 8.8.8.8: 64 bytes', b''))

            # テスト対象の関数を実行
            result = await ping(cmd_args)

            # コマンドが正しく実行されたかを確認
            mock_proc.assert_called_once_with(
                *cmd_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            # 結果が標準出力の内容と一致するか確認
            assert result == 'PING 8.8.8.8: 64 bytes'

    @pytest.mark.asyncio
    async def test_ping_stderr(self):
        """pingコマンドが失敗し、標準エラーを返す場合のテスト"""
        cmd_args = ['ping', '-c', '1', 'invalid.host']

        # 非同期プロセスをモックする
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_proc:
            mock_proc.return_value.communicate = AsyncMock(return_value=(b'', b'ping: cannot resolve invalid.host'))

            # テスト対象の関数を実行
            result = await ping(cmd_args)

            # コマンドが正しく実行されたかを確認
            mock_proc.assert_called_once_with(
                *cmd_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            # 結果が標準エラーの内容と一致するか確認
            assert result == 'ping: cannot resolve invalid.host'

    @pytest.mark.asyncio
    async def test_ping_no_output(self):
        """pingコマンドが出力を返さない場合のテスト"""
        cmd_args = ['ping', '-c', '1', '8.8.8.8']

        # 非同期プロセスをモックする
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_proc:
            mock_proc.return_value.communicate = AsyncMock(return_value=(b'', b''))

            # テスト対象の関数を実行
            result = await ping(cmd_args)

            # コマンドが正しく実行されたかを確認
            mock_proc.assert_called_once_with(
                *cmd_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            # 結果が空文字列であることを確認
            assert result == ''

    @pytest.mark.asyncio
    async def test_ping_partial_output(self):
        """pingコマンドが部分的に標準出力と標準エラーの両方を返す場合のテスト"""
        cmd_args = ['ping', '-c', '1', '8.8.8.8']

        # 非同期プロセスをモックする
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_proc:
            mock_proc.return_value.communicate = AsyncMock(return_value=(b'PING 8.8.8.8', b'Network error'))

            # テスト対象の関数を実行
            result = await ping(cmd_args)

            # コマンドが正しく実行されたかを確認
            mock_proc.assert_called_once_with(
                *cmd_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            # 標準出力が優先されて結果が返されることを確認
            assert result == 'PING 8.8.8.8'

class TestEvalutePingResult:      
    @pytest.mark.parametrize("packet_loss, avg_rtt, average_rtt, max_packet_loss, expected", [
        (0, 50, 100, 20, True),    # パケットロスとRTTが閾値以下 -> 成功
        (15, 90, 100, 20, True),   # パケットロスとRTTが閾値以下 -> 成功
        (25, 90, 100, 20, False),  # パケットロスが閾値を超えている -> 失敗
        (15, 120, 100, 20, False), # RTTが閾値を超えている -> 失敗
        (25, 120, 100, 20, False), # パケットロスとRTT両方が閾値を超えている -> 失敗
    ])
    def test_evaluate_ping_result(self, packet_loss, avg_rtt, average_rtt, max_packet_loss, expected):
        result = evaluate_ping_result(packet_loss, avg_rtt, average_rtt, max_packet_loss)
        assert result == expected, f"Test failed for packet_loss={packet_loss}, avg_rtt={avg_rtt}"

class TestValidatePingWindows:

    # Packet loss and RTT parser mocks (for simplicity)
    @pytest.fixture
    def mock_parse_packet_loss(self):
        with patch('script.parse_packet_loss') as mock:
            yield mock

    @pytest.fixture
    def mock_parse_rtt(self):
        with patch('script.parse_rtt') as mock:
            yield mock

    # 正常系テスト: 日本語版で "ng" ステータス時のエラーメッセージ
    @pytest.mark.parametrize("is_ja, output", [
        (True, "要求がタイムアウトしました"),
        (True, "宛先ホストに到達できません"),
        (False, "Request timed out"),
        (False, "Destination host unreachable")
    ])
    def test_ng_status(self, is_ja, output):
        assert validate_ping_windows('ng', is_ja, output, 100, 0) is True

    # 正常系テスト: 正常なパケットロスとRTTの解析（日本語版）
    @pytest.mark.parametrize("is_ja, output, packet_loss, avg_rtt", [
        (True, "パケットの損失 0% 平均 = 50ms", 0, 50),
        (False, "0% loss, Average = 50ms", 0, 50)
    ])
    def test_valid_rtt_loss(self, is_ja, output, packet_loss, avg_rtt, mock_parse_packet_loss, mock_parse_rtt):
        mock_parse_packet_loss.return_value = packet_loss
        mock_parse_rtt.return_value = avg_rtt
        assert validate_ping_windows('ok', is_ja, output, 100, 0) is True

    # 異常系テスト: RTTが取得できない、またはパケットロスが取得できない場合
    @pytest.mark.parametrize("is_ja, output, packet_loss, avg_rtt", [
        (True, "不正なパターン", None, None),
        (False, "Invalid pattern", None, None)
    ])
    def test_invalid_rtt_loss(self, is_ja, output, packet_loss, avg_rtt, mock_parse_packet_loss, mock_parse_rtt):
        mock_parse_packet_loss.return_value = packet_loss
        mock_parse_rtt.return_value = avg_rtt
        assert validate_ping_windows('ok', is_ja, output, 100, 0) is False

    # 異常系テスト: パケットロスが最大許容範囲を超える場合
    def test_exceed_packet_loss(self, mock_parse_packet_loss, mock_parse_rtt):
        mock_parse_packet_loss.return_value = 50  # 許容パケットロス超え
        mock_parse_rtt.return_value = 50
        assert validate_ping_windows('ok', False, "0% loss, Average = 50ms", 100, 20) is False

    # 正常系テスト: RTTが最大許容範囲内である場合
    def test_rtt_within_limit(self, mock_parse_packet_loss, mock_parse_rtt):
        mock_parse_packet_loss.return_value = 0
        mock_parse_rtt.return_value = 50  # RTTが許容範囲内
        assert validate_ping_windows('ok', False, "0% loss, Average = 50ms", 100, 0) is True

class TestValidatePingMac:

    # Packet loss and RTT parser mocks (for simplicity)
    @pytest.fixture
    def mock_parse_packet_loss(self):
        with patch('script.parse_packet_loss') as mock:
            yield mock

    @pytest.fixture
    def mock_parse_rtt(self):
        with patch('script.parse_rtt') as mock:
            yield mock

    # 正常系テスト: expected_statusが "ng" の場合、タイムアウトや到達不可のエラーメッセージがあるかをチェック
    @pytest.mark.parametrize("output", [
        "Request timeout for icmp_seq 0",
        "Destination Host Unreachable"
    ])
    def test_ng_status(self, output):
        assert validate_ping_mac('ng', output, 100, 20) is True

    # 異常系テスト: expected_statusが "ng" だが、エラーメッセージがない場合
    def test_ng_status_no_error(self):
        output = "64 bytes from 8.8.8.8: icmp_seq=0 ttl=57 time=10.2 ms"
        assert validate_ping_mac('ng', output, 100, 20) is False

    # 正常系テスト: パケットロスとRTTが取得でき、正常に判定されるケース
    @pytest.mark.parametrize("output, packet_loss, avg_rtt", [
        ("10 packets transmitted, 10 packets received, 0.0% packet loss, time 9015ms\n"
         "round-trip min/avg/max/stddev = 10.2/15.4/20.3/3.2 ms", 0, 15.4),
        ("10 packets transmitted, 10 packets received, 20.0% packet loss, time 9015ms\n"
         "round-trip min/avg/max/stddev = 50.5/75.0/100.3/10.2 ms", 20, 75.0)
    ])
    def test_valid_rtt_loss(self, output, packet_loss, avg_rtt, mock_parse_packet_loss, mock_parse_rtt):
        mock_parse_packet_loss.return_value = packet_loss
        mock_parse_rtt.return_value = avg_rtt
        assert validate_ping_mac('ok', output, 100, 20) is True

    # 異常系テスト: パケットロスやRTTが取得できない場合
    @pytest.mark.parametrize("output", [
        "Invalid ping output with no packet loss information",
        "round-trip min/avg/max/stddev = not available"
    ])
    def test_invalid_rtt_loss(self, output, mock_parse_packet_loss, mock_parse_rtt):
        mock_parse_packet_loss.return_value = None
        mock_parse_rtt.return_value = None
        assert validate_ping_mac('ok', output, 100, 0) is False

    # 異常系テスト: 許容される最大パケットロスを超えるケース
    def test_exceed_packet_loss(self, mock_parse_packet_loss, mock_parse_rtt):
        output = ("10 packets transmitted, 8 packets received, 30.0% packet loss, time 9015ms\n"
                  "round-trip min/avg/max/stddev = 10.2/15.4/20.3/3.2 ms")
        mock_parse_packet_loss.return_value = 30  # 許容パケットロス超え
        mock_parse_rtt.return_value = 15.4
        assert validate_ping_mac('ok', output, 100, 20) is False

    # 正常系テスト: RTTが最大許容範囲内であるケース
    def test_rtt_within_limit(self, mock_parse_packet_loss, mock_parse_rtt):
        output = ("10 packets transmitted, 10 packets received, 0.0% packet loss, time 9015ms\n"
                  "round-trip min/avg/max/stddev = 10.2/15.4/20.3/3.2 ms")
        mock_parse_packet_loss.return_value = 0
        mock_parse_rtt.return_value = 15.4  # RTTが許容範囲内
        assert validate_ping_mac('ok', output, 100, 0) is True

class TestValidatePing:
    
    @pytest.mark.parametrize("is_win, is_ja, average_rtt, max_packet_loss, output, expected_status, expected_result", [
        # Test case 1: expected_status is 'ng' and output indicates failure (timeout)
        (True, True, 100, 0, "要求がタイムアウトしました", 'ng', True),
        # Test case 2: expected_status is 'ng' and output does not indicate failure
        (True, True, 100, 0, "Reply from 8.8.8.8: bytes=32 time=50ms TTL=56", 'ng', False),
        # Test case 3: expected_status is 'ok' and Windows ping passes (average_rtt, max_packet_loss under limit)
        (True, False, 100, 0, "0% loss, Average = 50ms", 'ok', True),
        # Test case 4: expected_status is 'ok' but ping fails on Windows (timeout)
        (True, False, 100, 0, "Request timed out", 'ok', False),
        # Test case 5: expected_status is 'ok' and Mac ping passes (average_rtt, max_packet_loss under limit)
        (False, False, 100, 0, "10 packets transmitted, 10 packets received, 0.0% packet loss, time 9015ms\n"
         "round-trip min/avg/max/stddev = 10.2/15.4/20.3/3.2 ms", 'ok', True),
        # Test case 6: expected_status is 'ng' and Mac ping fails (unreachable)
        (False, False, 100, 0, "Destination Host Unreachable", 'ng', True),
        # Test case 7: output is None
        (False, False, 100, 0, None, 'ok', False)
    ])
    def test_validate_ping(self, is_win, is_ja, average_rtt, max_packet_loss, output, expected_status, expected_result):
        """
        validate_ping 関数の出力が期待される結果と一致するかどうかを確認するテスト
        """
        # validate_ping 関数の結果が期待通りかどうかを確認
        result = validate_ping(is_win, is_ja, average_rtt, max_packet_loss, output, expected_status)
        assert result == expected_result

class TestPingAndValidate:

    @pytest.fixture
    def mock_ping(self):
        with patch('script.ping', new_callable=AsyncMock) as mock:
            yield mock

    @pytest.fixture
    def mock_validate_ping(self):
        with patch('script.validate_ping') as mock:
            yield mock

    # Windowsでの成功テスト
    @pytest.mark.asyncio
    async def test_ping_and_validate_success_windows(self, mock_ping, mock_validate_ping):
        mock_ping.return_value = "Ping output"  # モックしたpingの出力
        mock_validate_ping.return_value = True  # pingが成功する場合

        output, success = await ping_and_validate(True, False, '4', 100, 10, "localhost", "ok")

        assert output == "Ping output"
        assert success is True
        mock_ping.assert_called_once_with(['ping', '-n', '4', 'localhost'])
        mock_validate_ping.assert_called_once_with(True, False, 100, 10, "Ping output", "ok")

    # Macでの成功テスト
    @pytest.mark.asyncio
    async def test_ping_and_validate_success_mac(self, mock_ping, mock_validate_ping):
        mock_ping.return_value = "Ping output"  # モックしたpingの出力
        mock_validate_ping.return_value = True  # pingが成功する場合

        output, success = await ping_and_validate(False, False, '4', 100, 10, "localhost", "ok")

        assert output == "Ping output"
        assert success is True
        mock_ping.assert_called_once_with(['ping', '-c', '4', 'localhost'])
        mock_validate_ping.assert_called_once_with(False, False, 100, 10, "Ping output", "ok")

    # Windowsでの失敗テスト
    @pytest.mark.asyncio
    async def test_ping_and_validate_failure_windows(self, mock_ping, mock_validate_ping):
        mock_ping.return_value = "Ping output"  # モックしたpingの出力
        mock_validate_ping.return_value = False  # pingが失敗する場合

        output, success = await ping_and_validate(True, False, '4', 100, 10, "localhost", "ok")

        assert output == "Ping output"
        assert success is False
        mock_ping.assert_called_once_with(['ping', '-n', '4', 'localhost'])
        mock_validate_ping.assert_called_once_with(True, False, 100, 10, "Ping output", "ok")

    # Macでの失敗テスト
    @pytest.mark.asyncio
    async def test_ping_and_validate_failure_mac(self, mock_ping, mock_validate_ping):
        mock_ping.return_value = "Ping output"  # モックしたpingの出力
        mock_validate_ping.return_value = False  # pingが失敗する場合

        output, success = await ping_and_validate(False, False, '4', 100, 10, "localhost", "ok")

        assert output == "Ping output"
        assert success is False
        mock_ping.assert_called_once_with(['ping', '-c', '4', 'localhost'])
        mock_validate_ping.assert_called_once_with(False, False, 100, 10, "Ping output", "ok")

    # pingの出力がNoneの場合
    @pytest.mark.asyncio
    async def test_ping_and_validate_none_output(self, mock_ping, mock_validate_ping):
        mock_ping.return_value = None  # pingの出力がNone
        mock_validate_ping.return_value = False  # バリデーションもFalse

        output, success = await ping_and_validate(True, False, '4', 100, 10, "localhost", "ok")

        assert output is None
        assert success is False
        mock_ping.assert_called_once_with(['ping', '-n', '4', 'localhost'])
        mock_validate_ping.assert_called_once_with(True, False, 100, 10, None, "ok")


class TestProcessPing:

    @pytest.mark.asyncio
    async def test_process_ping_success(self):
        # テスト用の入力データ
        is_win = True
        is_ja = False
        ping_count = '4'
        average_rtt = 100
        max_packet_loss = 10
        kyoten_name = 'Test Kyoten'
        test_type = 'Test Type'
        index = 0
        host = 'localhost'
        expected_status = 'ok'
        results_dir = './results'

        # モック設定
        with mock.patch('script.ping_and_validate', return_value=('Ping output', True)) as mock_ping_and_validate, \
             mock.patch('script.save_results') as mock_save_results:

            # 関数の実行
            result = await process_ping(is_win, is_ja, ping_count, average_rtt, max_packet_loss, kyoten_name, test_type, index, host, expected_status, results_dir)

            # 期待される呼び出しを確認
            mock_ping_and_validate.assert_called_once_with(is_win, is_ja, ping_count, average_rtt, max_packet_loss, host, expected_status)
            mock_save_results.assert_called_once_with(kyoten_name, test_type, index, host, 'Ping output', results_dir, 'ping')

            # 戻り値の確認
            assert result == {host: True}

    @pytest.mark.asyncio
    async def test_process_ping_failure(self):
        # テスト用の入力データ
        is_win = True
        is_ja = False
        ping_count = '4'
        average_rtt = 100
        max_packet_loss = 10
        kyoten_name = 'Test Kyoten'
        test_type = 'Test Type'
        index = 0
        host = 'localhost'
        expected_status = 'ng'
        results_dir = './results'

        # モック設定
        with mock.patch('script.ping_and_validate', return_value=('Ping output', False)) as mock_ping_and_validate, \
             mock.patch('script.save_results') as mock_save_results:

            # 関数の実行
            result = await process_ping(is_win, is_ja, ping_count, average_rtt, max_packet_loss, kyoten_name, test_type, index, host, expected_status, results_dir)

            # 期待される呼び出しを確認
            mock_ping_and_validate.assert_called_once_with(is_win, is_ja, ping_count, average_rtt, max_packet_loss, host, expected_status)
            mock_save_results.assert_called_once_with(kyoten_name, test_type, index, host, 'Ping output', results_dir, 'ping')

            # 戻り値の確認
            assert result == {host: False}

@pytest.mark.asyncio
class TestPingMultipleHosts:

    async def test_ping_multiple_hosts_success(self):
        is_win = True
        is_ja = False
        ping_count = '4'
        average_rtt = 100
        max_packet_loss = 10
        kyoten_name = 'Test Kyoten'
        test_type = 'Test Type'
        results_dir = './results'

        # テスト用のリストデータ
        list_ping_eval = [
            {'localhost': 'ok'},
            {'example.com': 'ng'}
        ]

        # モック設定
        with mock.patch('script.process_ping') as mock_process_ping:
            # モックの戻り値設定
            mock_process_ping.side_effect = [
                {'localhost': True},  # 1つ目の呼び出し
                {'example.com': False}  # 2つ目の呼び出し
            ]

            # 関数の実行
            result = await ping_multiple_hosts(is_win, is_ja, ping_count, average_rtt, max_packet_loss, kyoten_name, test_type, list_ping_eval, results_dir)

            # モックが期待通りに呼び出されたかを確認
            assert mock_process_ping.call_count == 2

            # 呼び出し順序の確認
            mock_process_ping.assert_any_call(is_win, is_ja, ping_count, average_rtt, max_packet_loss, kyoten_name, test_type, 1, 'localhost', 'ok', results_dir)
            mock_process_ping.assert_any_call(is_win, is_ja, ping_count, average_rtt, max_packet_loss, kyoten_name, test_type, 2, 'example.com', 'ng', results_dir)

            # 戻り値の確認
            assert result == [
                {'localhost': True},
                {'example.com': False}
            ]

    async def test_ping_multiple_hosts_empty_list(self):
        is_win = True
        is_ja = False
        ping_count = '4'
        average_rtt = 100
        max_packet_loss = 10
        kyoten_name = 'Test Kyoten'
        test_type = 'Test Type'
        results_dir = './results'

        # 空のリストを使用
        list_ping_eval = []

        with mock.patch('script.process_ping') as mock_process_ping:
            result = await ping_multiple_hosts(is_win, is_ja, ping_count, average_rtt, max_packet_loss, kyoten_name, test_type, list_ping_eval, results_dir)

            # モックが呼び出されないことを確認
            mock_process_ping.assert_not_called()

            # 戻り値は空のリストであることを確認
            assert result == []

# IPアドレス抽出関数のテスト
def test_extract_ips():
    trace_result = [
        "1 192.168.1.1 1.123 ms",
        "2 10.0.0.1 5.456 ms",
        "3 8.8.8.8 2.789 ms"
    ]
    expected_ips = ['192.168.1.1', '10.0.0.1', '8.8.8.8']
    
    assert extract_ips(trace_result) == expected_ips

# ルート一致チェック関数のテスト
def test_check_route_match():
    list_ip = ['192.168.1.1', '10.0.0.1', '8.8.8.8']
    list_expected_route = ['10.0.0.1', '8.8.4.4', '1.1.1.1']
    
    assert check_route_match(list_ip, list_expected_route) == True

    list_expected_route = ['9.9.9.9']
    assert check_route_match(list_ip, list_expected_route) == False

class TestValidateRoute:

    def test_validate_route_success(self):
        trace_output = """
        traceroute to google.com (8.8.8.8), 30 hops max
        1  192.168.1.1  1.123 ms
        2  10.0.0.1  5.456 ms
        3  8.8.8.8  2.789 ms
        """
        expected_route = ['192.168.1.1', '10.0.0.1', '8.8.8.8']
        
        assert validate_route(trace_output, expected_route) == True

    def test_validate_route_failure(self):
        trace_output = """
        traceroute to google.com (8.8.8.8), 30 hops max
        1  192.168.1.1  1.123 ms
        2  10.0.0.1  5.456 ms
        3  8.8.8.8  2.789 ms
        """
        expected_route = ['9.9.9.9']
        
        assert validate_route(trace_output, expected_route) == False

@pytest.mark.asyncio
class TestTraceroute:

    @mock.patch('asyncio.create_subprocess_exec')
    async def test_traceroute_success(self, mock_create_subprocess_exec):
        # モックされたプロセスの設定
        mock_process = mock.AsyncMock()
        mock_process.communicate.return_value = (b'1  192.168.1.1  1.123 ms\n2  10.0.0.1  5.456 ms\n3  8.8.8.8  2.789 ms\n', b'')
        mock_create_subprocess_exec.return_value = mock_process

        # コマンド引数
        cmd_args = ['traceroute', 'example.com']

        # 関数を呼び出す
        result = await traceroute(cmd_args)

        # 戻り値の確認
        assert result == '1  192.168.1.1  1.123 ms\n2  10.0.0.1  5.456 ms\n3  8.8.8.8  2.789 ms\n'

    @mock.patch('asyncio.create_subprocess_exec')
    async def test_traceroute_error(self, mock_create_subprocess_exec):
        # モックされたプロセスの設定（エラー出力）
        mock_process = mock.AsyncMock()
        mock_process.communicate.return_value = (b'', b'Error: Unable to reach host')
        mock_create_subprocess_exec.return_value = mock_process

        # コマンド引数
        cmd_args = ['traceroute', 'nonexistent.host']

        # 関数を呼び出す
        result = await traceroute(cmd_args)

        # エラー出力が戻ることを確認
        assert result == ''

        # エラーメッセージを確認したい場合は、stderrを確認する必要があります
        assert mock_process.communicate.call_count == 1 

@pytest.mark.asyncio
class TestTraceAndValidate:

    @mock.patch('script.traceroute')  # traceroute 関数をモック
    @mock.patch('script.validate_route')  # validate_route 関数をモック
    async def test_trace_and_validate_success_win(self, mock_validate_route, mock_traceroute):
        # モックされたトレース出力
        mock_traceroute.return_value = "1  192.168.1.1  1.123 ms\n2  10.0.0.1  5.456 ms\n3  8.8.8.8  2.789 ms\n"
        mock_validate_route.return_value = True  # 期待される経路に一致

        is_win = True
        max_hop = 10
        host = 'localhost'
        list_expected_route = ['192.168.1.1', '10.0.0.1', '8.8.8.8']

        # 関数を呼び出す
        result = await trace_and_validate(is_win, max_hop, host, list_expected_route)

        # 戻り値の確認
        assert result == (mock_traceroute.return_value, True)

        # 呼び出しの確認
        mock_traceroute.assert_called_once_with(['tracert', '-h 10', host])
        mock_validate_route.assert_called_once_with(mock_traceroute.return_value, list_expected_route)

    @mock.patch('script.traceroute')  # traceroute 関数をモック
    @mock.patch('script.validate_route')  # validate_route 関数をモック
    async def test_trace_and_validate_failure_win(self, mock_validate_route, mock_traceroute):
        # モックされたトレース出力
        mock_traceroute.return_value = "1  192.168.1.1  1.123 ms\n2  10.0.0.1  5.456 ms\n3  8.8.8.8  2.789 ms\n"
        mock_validate_route.return_value = False  # 期待される経路に一致しない

        is_win = True
        max_hop = 10
        host = 'localhost'
        list_expected_route = ['192.168.1.1', '10.0.0.1', '1.1.1.1']  # 不一致

        # 関数を呼び出す
        result = await trace_and_validate(is_win, max_hop, host, list_expected_route)

        # 戻り値の確認
        assert result == (mock_traceroute.return_value, False)

        # 呼び出しの確認
        mock_traceroute.assert_called_once_with(['tracert', '-h 10', host])
        mock_validate_route.assert_called_once_with(mock_traceroute.return_value, list_expected_route)

    @mock.patch('script.traceroute')  # traceroute 関数をモック
    @mock.patch('script.validate_route')  # validate_route 関数をモック
    async def test_trace_and_validate_success_mac(self, mock_validate_route, mock_traceroute):
        # モックされたトレース出力
        mock_traceroute.return_value = "traceroute to 8.8.8.8 (8.8.8.8), 30 hops max\n1  192.168.1.1  1.123 ms\n2  10.0.0.1  5.456 ms\n3  8.8.8.8  2.789 ms\n"
        mock_validate_route.return_value = True  # 期待される経路に一致

        is_win = False
        max_hop = 10
        host = '8.8.8.8'
        list_expected_route = ['192.168.1.1', '10.0.0.1', '8.8.8.8']

        # 関数を呼び出す
        result = await trace_and_validate(is_win, max_hop, host, list_expected_route)

        # 戻り値の確認
        assert result == (mock_traceroute.return_value, True)

        # 呼び出しの確認
        mock_traceroute.assert_called_once_with(['traceroute', '-n', '-m', '10', host])
        mock_validate_route.assert_called_once_with(mock_traceroute.return_value, list_expected_route)

    @mock.patch('script.traceroute')  # traceroute 関数をモック
    @mock.patch('script.validate_route')  # validate_route 関数をモック
    async def test_trace_and_validate_failure_mac(self, mock_validate_route, mock_traceroute):
        # モックされたトレース出力
        mock_traceroute.return_value = "traceroute to 8.8.8.8 (8.8.8.8), 30 hops max\n1  192.168.1.1  1.123 ms\n2  10.0.0.1  5.456 ms\n3  8.8.8.8  2.789 ms\n"
        mock_validate_route.return_value = False  # 期待される経路に一致しない

        is_win = False
        max_hop = 10
        host = '8.8.8.8'
        list_expected_route = ['192.168.1.1', '10.0.0.1', '1.1.1.1']  # 不一致

        # 関数を呼び出す
        result = await trace_and_validate(is_win, max_hop, host, list_expected_route)

        # 戻り値の確認
        assert result == (mock_traceroute.return_value, False)

        # 呼び出しの確認
        mock_traceroute.assert_called_once_with(['traceroute', '-n', '-m', '10', host])
        mock_validate_route.assert_called_once_with(mock_traceroute.return_value, list_expected_route)


@pytest.mark.asyncio
class TestProcessTrace:

    @mock.patch('script.trace_and_validate')  # trace_and_validate関数をモック
    @mock.patch('script.save_results')  # save_results関数をモック
    async def test_process_trace_success(self, mock_save_results, mock_trace_and_validate):
        # モックされたトレース出力
        mock_trace_and_validate.return_value = ("traceroute output", True)

        is_win = True
        max_hop = 10
        kyoten_name = 'Tokyo'
        test_type = 'PingTest'
        index = 1
        host = 'localhost'
        list_expected_route = ['192.168.1.1', '10.0.0.1', '8.8.8.8']
        results_dir = '/path/to/results'

        # 関数を呼び出す
        result = await process_trace(is_win, max_hop, kyoten_name, test_type, index, host, list_expected_route, results_dir)

        # 戻り値の確認
        assert result == {host: True}

        # 呼び出しの確認
        mock_trace_and_validate.assert_called_once_with(is_win, max_hop, host, list_expected_route)
        mock_save_results.assert_called_once_with(kyoten_name, test_type, index, host, "traceroute output", results_dir, "traceroute")

    @mock.patch('script.trace_and_validate')  # trace_and_validate関数をモック
    @mock.patch('script.save_results')  # save_results関数をモック
    async def test_process_trace_failure(self, mock_save_results, mock_trace_and_validate):
        # モックされたトレース出力
        mock_trace_and_validate.return_value = ("traceroute output", False)

        is_win = True
        max_hop = 10
        kyoten_name = 'Tokyo'
        test_type = 'PingTest'
        index = 1
        host = 'localhost'
        list_expected_route = ['192.168.1.1', '10.0.0.1', '8.8.8.8']
        results_dir = '/path/to/results'

        # 関数を呼び出す
        result = await process_trace(is_win, max_hop, kyoten_name, test_type, index, host, list_expected_route, results_dir)

        # 戻り値の確認
        assert result == {host: False}

        # 呼び出しの確認
        mock_trace_and_validate.assert_called_once_with(is_win, max_hop, host, list_expected_route)
        mock_save_results.assert_called_once_with(kyoten_name, test_type, index, host, "traceroute output", results_dir, "traceroute")
        
@pytest.mark.asyncio
class TestTraceMultipleHosts:

    @mock.patch('script.process_trace')  # process_trace関数をモック
    async def test_trace_multiple_hosts_success(self, mock_process_trace):
        # モックされたトレース出力
        mock_process_trace.side_effect = [
            { 'localhost': True }, 
            { 'example.com': True }
        ]

        is_win = True
        max_hop = 10
        kyoten_name = 'Tokyo'
        test_type = 'TraceTest'
        list_trace_eval = [
            {'localhost': ['192.168.1.1', '10.0.0.1']},
            {'example.com': ['8.8.8.8']}
        ]
        results_dir = '/path/to/results'

        # 関数を呼び出す
        results = await trace_multiple_hosts(is_win, max_hop, kyoten_name, test_type, list_trace_eval, results_dir)

        # 戻り値の確認
        assert results == [{ 'localhost': True }, { 'example.com': True }]

        # 呼び出しの確認
        assert mock_process_trace.call_count == 2  # 2回呼び出されることを確認

    @mock.patch('script.process_trace')  # process_trace関数をモック
    async def test_trace_multiple_hosts_failure(self, mock_process_trace):
        # モックされたトレース出力
        mock_process_trace.side_effect = [
            { 'localhost': True }, 
            { 'example.com': False }
        ]

        is_win = True
        max_hop = 10
        kyoten_name = 'Tokyo'
        test_type = 'TraceTest'
        list_trace_eval = [
            {'localhost': ['192.168.1.1', '10.0.0.1']},
            {'example.com': ['8.8.8.8']}
        ]
        results_dir = '/path/to/results'

        # 関数を呼び出す
        results = await trace_multiple_hosts(is_win, max_hop, kyoten_name, test_type, list_trace_eval, results_dir)

        # 戻り値の確認
        assert results == [{ 'localhost': True }, { 'example.com': False }]

        # 呼び出しの確認
        assert mock_process_trace.call_count == 2  # 2回呼び出されることを確認

@pytest.fixture
def mock_os_functions():
    """
    os.path.exists と os.makedirs をモックするためのフィクスチャ
    """
    with patch('os.path.exists') as mock_exists, patch('os.makedirs') as mock_makedirs:
        yield mock_exists, mock_makedirs

@pytest.fixture
def mock_ping_and_trace_functions():
    """
    ping_multiple_hosts と trace_multiple_hosts をモックするためのフィクスチャ
    """
    with patch('script.ping_multiple_hosts', new_callable=AsyncMock) as mock_ping, \
         patch('script.trace_multiple_hosts', new_callable=AsyncMock) as mock_trace:
        yield mock_ping, mock_trace

@pytest.fixture
def mock_create_unique_folder():
    """
    create_unique_folder 関数をモックするためのフィクスチャ
    """
    with patch('script.create_unique_folder') as mock_create_folder:
        yield mock_create_folder

@pytest.mark.asyncio
class TestPingTraceMultipleHosts:
    """
    ping_trace_multiple_hosts 関数に対するテストクラス
    """

    @pytest.fixture(autouse=True)
    def setup(self, mock_os_functions, mock_ping_and_trace_functions, mock_create_unique_folder):
        """
        テストクラスのセットアップとしてモックを適用
        """
        self.mock_ping, self.mock_trace = mock_ping_and_trace_functions
        self.mock_create_unique_folder = mock_create_unique_folder

        # モック関数の戻り値を定義
        self.mock_ping.return_value = {'host1': 'success', 'host2': 'failure'}
        self.mock_trace.return_value = {'host1': 'trace_success', 'host2': 'trace_failure'}
        self.mock_create_unique_folder.return_value = '/dummy/results/unique_folder'

        # テスト用のデフォルト引数
        self.is_win = True
        self.is_ja = False
        self.ping_count = 4
        self.average_rtt = 100
        self.max_packet_loss = 10
        self.max_hop = 30
        self.list_ping_eval = ['host1', 'host2']
        self.list_trace_eval = ['host1', 'host2']
        self.selected_kyoten_name = 'kyoten_test'
        self.selected_test_type = 'ping_trace_test'

    async def test_ping_trace_multiple_hosts_success(self):
        """
        ping_trace_multiple_hosts 関数が正常に動作するかのテスト
        """
        # 関数の実行
        ping_results, trace_results, results_dir = await ping_trace_multiple_hosts(
            self.is_win, self.is_ja, self.ping_count, self.average_rtt, self.max_packet_loss, 
            self.max_hop, self.list_ping_eval, self.list_trace_eval, 
            self.selected_kyoten_name, self.selected_test_type
        )

        # 正しい結果が返されるかのアサーション
        assert ping_results == {'host1': 'success', 'host2': 'failure'}
        assert trace_results == {'host1': 'trace_success', 'host2': 'trace_failure'}
        assert results_dir == '/dummy/results/unique_folder'

        # モックが期待通りに呼ばれているか確認
        self.mock_ping.assert_called_once_with(
            self.is_win, self.is_ja, self.ping_count, self.average_rtt, self.max_packet_loss, 
            self.selected_kyoten_name, self.selected_test_type, self.list_ping_eval, results_dir
        )
        self.mock_trace.assert_called_once_with(
            self.is_win, self.max_hop, self.selected_kyoten_name, 
            self.selected_test_type, self.list_trace_eval, results_dir
        )
        self.mock_create_unique_folder.assert_called_once_with(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'results', self.selected_kyoten_name), 
            self.selected_test_type
        )
        
def test_convert_to_list():
    # テスト用のデータフレームを作成
    data = {
        'dest': ['8.8.8.8', '1.1.1.1'],
        'ping_eval': ['ok', 'timeout'],
        'trace_eval_1': ['192.168.1.1', '192.168.1.1'],
        'trace_eval_2': ['10.0.0.1', '10.0.0.1'],
        'trace_eval_3': ['8.8.8.8', '1.1.1.1']
    }
    df = pd.DataFrame(data)

    # 期待される結果を設定
    expected_ping_eval = [
        {'8.8.8.8': 'ok'},
        {'1.1.1.1': 'timeout'}
    ]
    
    expected_trace_eval = [
        {'8.8.8.8': ['192.168.1.1', '10.0.0.1', '8.8.8.8']},
        {'1.1.1.1': ['192.168.1.1', '10.0.0.1', '1.1.1.1']}
    ]

    # convert_to_list関数を呼び出す
    list_ping_eval, list_trace_eval = convert_to_list(df)

    # 結果が期待通りであることを確認
    assert list_ping_eval == expected_ping_eval
    assert list_trace_eval == expected_trace_eval
    
# テスト用のDataFrameを作成
@pytest.fixture
def sample_df():
    data = {
        'number': [1, 2, 3],
        'area': ['エリアA', 'エリアB', 'エリアC'],
        'name': ['拠点A', '拠点B', '拠点C'],
        'type': ['タイプ1', 'タイプ2', 'タイプ3']
    }
    return pd.DataFrame(data)

class TestSelectKyoten:

    def test_select_kyoten_valid_input(self, sample_df):
        user_input = '1'  # 正しい入力
        with patch('builtins.input', return_value=user_input):
            selected_name, selected_type = select_kyoten(sample_df)

            assert selected_name == '拠点A'
            assert selected_type == 'タイプ1'

    def test_select_kyoten_invalid_input(self, sample_df):
        user_inputs = ['4', 'invalid', '2']  # 最初は無効、次に無効、最後に有効な入力
        with patch('builtins.input', side_effect=user_inputs):
            selected_name, selected_type = select_kyoten(sample_df)

            assert selected_name == '拠点B'
            assert selected_type == 'タイプ2'

    def test_select_kyoten_multiple_invalid_input(self, sample_df):
        user_inputs = ['invalid', 'invalid', '3']  # 全て無効、最後に有効な入力
        with patch('builtins.input', side_effect=user_inputs):
            selected_name, selected_type = select_kyoten(sample_df)

            assert selected_name == '拠点C'
            assert selected_type == 'タイプ3'
            
# テスト用のCSVファイルの内容を定義
TEST_CSV_CONTENT = """kyoten_type,test_type
Tokyo,Test1
Tokyo,Test2
"""

@pytest.fixture
def mock_csv_file(tmpdir):
    """モック用のCSVファイルを作成するフィクスチャ"""
    csv_file = tmpdir.join("test_info.csv")
    csv_file.write(TEST_CSV_CONTENT)
    return str(csv_file)

class TestSelectTestType:
    
    def test_select_test_type_success(self, mock_csv_file, capsys):
        """成功パターンのテスト"""
        with mock.patch('builtins.input', side_effect=['1']):  # ユーザー入力をモック
            result = select_test_type('Tokyo', csv_file=mock_csv_file)
            
            # 出力を確認
            captured = capsys.readouterr()
            assert "試験内容を表示します:" in captured.out
            assert "選択された試験: Test1" in captured.out
            assert result == "Test1"
            
    def test_select_test_type_no_tests(self, mock_csv_file, capsys):
            """試験が設定されていないケースのテスト"""
            with mock.patch('builtins.input', side_effect=['1']):  # ユーザー入力をモック
                result = select_test_type('Osaka', csv_file=mock_csv_file)
                
                # 出力を確認
                captured = capsys.readouterr()
                assert "選択された拠点には試験が設定されていません。" in captured.out
                assert result is None

    def test_select_test_type_invalid_input(self, mock_csv_file, capsys):
        """無効な入力が与えられた場合のテスト"""
        with mock.patch('builtins.input', side_effect=['3', 'invalid', '1']):
            result = select_test_type('Tokyo', csv_file=mock_csv_file)
            
            # 出力を確認
            captured = capsys.readouterr()
            assert "選択したnumberは存在しません。もう一度選択してください。" in captured.out
            assert "無効な入力です。数字を入力してください。" in captured.out
            assert result == "Test1"  # 正しい入力が最後に来た場合の結果
    
    def test_select_test_type_file_not_found(self, capsys):
        """ファイルが見つからない場合のテスト"""
        result = select_test_type('Tokyo', csv_file='non_existent_file.csv')
        
        # 出力を確認
        captured = capsys.readouterr()
        assert "CSVファイルが見つかりません。" in captured.out
        assert result is None
            
# テスト用のCSVデータを用意
TEST_CSV_DATA = """
kyoten_type,test_type,file_name
大規模,正常性試験,test_type_a.csv
大規模,帯域保証網試験,test_type_b.csv
中規模,正常性試験,test_type_c.csv
小規模,復旧試験,test_type_d.csv
"""

class TestGetTestTypePath:

    @pytest.fixture
    def mock_test_info_csv(self, tmpdir):
        """テスト用のCSVファイルを一時的に作成するフィクスチャ"""
        test_info_path = tmpdir.join("test_info.csv")
        with open(test_info_path, 'w') as f:
            f.write(TEST_CSV_DATA)
        return str(test_info_path)

    def test_get_test_type_path_valid(self, mock_test_info_csv):
        """有効な選択肢に対して正しいファイルパスを返すことをテスト"""
        result_1 = get_test_type_path("大規模", "正常性試験", mock_test_info_csv)
        assert result_1 == "../settings/test_type_a.csv"
        result_2 = get_test_type_path("中規模", "正常性試験", mock_test_info_csv)
        assert result_2 == "../settings/test_type_c.csv"

    def test_get_test_type_path_invalid_type(self, mock_test_info_csv):
        """無効な拠点タイプに対して例外が発生することをテスト"""
        with pytest.raises(SelectionError, match="選択された拠点タイプまたは試験タイプが見つかりませんでした。"):
            get_test_type_path("無効なタイプ", "正常性試験", mock_test_info_csv)

    def test_get_test_type_path_invalid_test(self, mock_test_info_csv):
        """無効な試験タイプに対して例外が発生することをテスト"""
        with pytest.raises(SelectionError, match="選択された拠点タイプまたは試験タイプが見つかりませんでした。"):
            get_test_type_path("大規模", "無効な試験", mock_test_info_csv)

    def test_get_test_type_path_invalid_both(self, mock_test_info_csv):
        """無効な拠点タイプと試験タイプに対して例外が発生することをテスト"""
        with pytest.raises(SelectionError, match="選択された拠点タイプまたは試験タイプが見つかりませんでした。"):
            get_test_type_path("無効なタイプ", "無効な試験", mock_test_info_csv)
        
class TestGenerateUniqueFilename:

    @mock.patch('os.path.exists')
    def test_generate_unique_filename_first_call(self, mock_exists):
        """
        最初のファイル生成時のテスト: ファイルが存在しない場合、インクリメントされないことを確認。
        """
        mock_exists.return_value = False  # ファイルが存在しない場合

        results_dir = '../files'
        selected_kyoten_name = 'kyoten_name'
        selected_test_type = 'test_type'

        expected_filename = f'{results_dir}/{selected_kyoten_name}_{selected_test_type}_判定表.xlsx'
        generated_filename = generate_unique_filename(results_dir, selected_kyoten_name, selected_test_type)

        assert generated_filename == expected_filename

    @mock.patch('os.path.exists')
    def test_generate_unique_filename_second_call(self, mock_exists):
        """
        2回目のファイル生成時のテスト: 同じ名前のファイルが存在する場合、(1)が付くことを確認。
        """
        # 1回目は存在しないが、2回目は存在する
        mock_exists.side_effect = [True, False]

        results_dir = '../files'
        selected_kyoten_name = 'kyoten_name'
        selected_test_type = 'test_type'

        expected_filename = f'{results_dir}/{selected_kyoten_name}_{selected_test_type}_判定表(1).xlsx'
        generated_filename = generate_unique_filename(results_dir, selected_kyoten_name, selected_test_type)

        assert generated_filename == expected_filename

    @mock.patch('os.path.exists')
    def test_generate_unique_filename_third_call(self, mock_exists):
        """
        3回目のファイル生成時のテスト: 連続してファイルが存在する場合、(2)が付くことを確認。
        """
        # 1回目と2回目は存在し、3回目は存在しない
        mock_exists.side_effect = [True, True, False]

        results_dir = '../files'
        selected_kyoten_name = 'kyoten_name'
        selected_test_type = 'test_type'

        expected_filename = f'{results_dir}/{selected_kyoten_name}_{selected_test_type}_判定表(2).xlsx'
        generated_filename = generate_unique_filename(results_dir, selected_kyoten_name, selected_test_type)

        assert generated_filename == expected_filename

@pytest.fixture
def mock_os_functions():
    """
    os.path.exists と os.makedirs をモックするためのフィクスチャ
    """
    with patch('os.path.exists') as mock_exists, patch('os.makedirs') as mock_makedirs:
        yield mock_exists, mock_makedirs
            
class TestCreateUniqueFolder:

    def test_create_unique_folder_first_time(self, mock_os_functions):
        """
        フォルダが初めて作成される場合のテスト
        """
        mock_exists, mock_makedirs = mock_os_functions

        # フォルダが存在しないことをシミュレート
        mock_exists.return_value = False

        base_dir = "/dummy/base"
        folder_name = "test_folder"

        # 関数を実行
        folder_path = create_unique_folder(base_dir, folder_name)

        # 正しいパスが返されることを確認
        expected_path = os.path.join(base_dir, folder_name)
        assert folder_path == expected_path

        # makedirs が一度だけ呼ばれていることを確認
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)


    def test_create_unique_folder_with_existing_folder(self, mock_os_functions):
        """
        フォルダが既に存在する場合、(1), (2), ... のように番号が付くかどうかのテスト
        """
        mock_exists, mock_makedirs = mock_os_functions

        # 最初のフォルダは存在するが、(1) は存在しないことをシミュレート
        mock_exists.side_effect = lambda path: path == os.path.join("/dummy/base", "test_folder")

        base_dir = "/dummy/base"
        folder_name = "test_folder"

        # 関数を実行
        folder_path = create_unique_folder(base_dir, folder_name)

        # 正しいパスが返されることを確認
        expected_path = os.path.join(base_dir, "test_folder(1)")
        assert folder_path == expected_path

        # makedirs が一度だけ呼ばれていることを確認
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)


    def test_create_unique_folder_with_multiple_existing_folders(self, mock_os_functions):
        """
        フォルダがすでに複数存在する場合、正しい番号が付くかどうかのテスト
        """
        mock_exists, mock_makedirs = mock_os_functions

        # フォルダが test_folder, test_folder(1), test_folder(2) まで存在する場合をシミュレート
        mock_exists.side_effect = lambda path: path in [
            os.path.join("/dummy/base", "test_folder"),
            os.path.join("/dummy/base", "test_folder(1)"),
            os.path.join("/dummy/base", "test_folder(2)")
        ]

        base_dir = "/dummy/base"
        folder_name = "test_folder"

        # 関数を実行
        folder_path = create_unique_folder(base_dir, folder_name)

        # 正しいパスが返されることを確認
        expected_path = os.path.join(base_dir, "test_folder(3)")
        assert folder_path == expected_path

        # makedirs が一度だけ呼ばれていることを確認
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)
  
class TestGetNameByHost:

    @pytest.fixture
    def df(self):
        data = {
            'dest': ['8.8.8.8', '8.8.4.4', '100.100.100.100'],
            'name': ['google_dns_1', 'google_dns_2', 'yahoo'],
            'data1': ['data_1', 'data_2', 'data_3'],
            'data2': ['data_1', 'data_2', 'data_3'],
            'data3': ['data_1', 'data_2', 'data_3'],
        }
        return pd.DataFrame(data)

    # 正常なホスト名でのテストケース
    def test_get_name_by_host_valid(self, df):
        assert get_name_by_host(df, '8.8.8.8') == 'google_dns_1'
        assert get_name_by_host(df, '100.100.100.100') == 'yahoo'

    # 存在しないホスト名でのテストケース
    def test_get_name_by_host_invalid(self, df):
        assert get_name_by_host(df, '1.1.1.1') is None

    # 空のホスト名でのテストケース
    def test_get_name_by_host_empty(self, df):
        assert get_name_by_host(df, '') is None

    # Noneホスト名でのテストケース
    def test_get_name_by_host_none(self, df):
        assert get_name_by_host(df, None) is None


class TestCheckOSAndLanguage:

    @patch('script.is_windows', return_value=True)
    @patch('script.is_japanese_os', return_value=True)
    @patch('script.is_english_os', return_value=False)
    def test_check_os_and_language_japanese(self, mock_is_english, mock_is_japanese, mock_is_windows):
        result = check_os_and_language()
        assert result == (True, True)

    @patch('script.is_windows', return_value=True)
    @patch('script.is_japanese_os', return_value=False)
    @patch('script.is_english_os', return_value=True)
    def test_check_os_and_language_english(self, mock_is_english, mock_is_japanese, mock_is_windows):
        result = check_os_and_language()
        assert result == (True, False)

    @patch('script.is_windows', return_value=True)
    @patch('script.is_japanese_os', return_value=False)
    @patch('script.is_english_os', return_value=False)
    def test_check_os_and_language_unsupported_language(self, mock_is_english, mock_is_japanese, mock_is_windows):
        with pytest.raises(UnsupportedLanguageError):
            check_os_and_language()

    @patch('script.is_mac', return_value=False)
    @patch('script.is_windows', return_value=False)
    def test_check_os_and_language_unsupported_os(self, mock_is_windows, mock_is_mac):
        with pytest.raises(UnsupportedOSError):
            check_os_and_language()