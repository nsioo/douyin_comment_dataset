# 抖音爬虫


## 介绍
这是一套全流程的抖音爬虫框架, 从移动端的模拟点击滑动, 到获取response的 url 和 cookies,  最后根据得到的url 和 cookies, 通过x-gorgon 加密算法, 进行抖音评论数据抓取.   
## 用法
1. 从网上下载`Android SDK Tools`, 地址:<https://developer.android.com/studio/releases/platform-tools>

2. 打开手机, 点击setting -> about phone -> software information -> build number, 连续点击 build number, 直到出现 start develper options, 在developer options 中找到USB debugging(USB 调试) 和 USB installation(通过USB安装) 并打开.

3.  点击 wifi ->当前wifi -> 找到IP addres  

4. 下载解压的`Android SDK Tools`, 在文件位置打开`cmd`, 运行`./adb start-server`, 使用usb连接电脑和手机, 输入`./adb devices`查看设备, 如果有设备, 输入`adb tcpip 5555` , 此步骤可以将手机转为无线连接, 当出现restarting in tcpip 5555 时表示操作成功,  此时断开手机与电脑的usb连接, 在电脑端输入`adb connect 第三步的手机IP:5555`就可以连接手机了, 此时查看`adb devices`应该显示`your phone IP:5555`

5. 打开TikTok, 运行`auto.py` 查看使用可能正常滑动和点击, 因为auto.py脚本通过点击抖音的固定位置来实现, 不同型号的手机可能大小不同, 手机点击的位置也可能不同, 所以请根据手机自行调整, 关于如何查看点击位置, 可以在手机的developer options 中打`show layout bounds` 查看点击位置


6. 下载`virtualxposed`框架, 地址:<https://github.com/android-hacker/VirtualXposed/releases>,下载`just trust me`模组:<https://github.com/Fuzion24/JustTrustMe/releases> 将两个apk文件放入手机, 在安装好virtualxposed 文件后, 点击virtualxposed, 导入刚才的下载的just trust me 文件, 并在xposed中的module中勾选, 在virtualxposed中选择重启, 将抖音移动至virtualxposed框架中, 点击运行, 如果运行说明xposed安装成功.


7. 运行`pip install mitmproxy` 安装mitmproxy, 通过`ipconfig`查看当前电脑IP地址, 打开手机 wifi -> 当前连接 -> 代理设为`手动` -> IP 设置为 电脑IP, 端口为`8080`, 点击保存,此时手机是无法上网的,  在手机浏览器中, 输入网页 `mitm.it`, 找到对应android图标, 下载安装mitmproxy证书并安装, 在电脑端运行`mitmpdmp -s ./mitm_script.py`, 在`virtualxposed`框架中打开TikTok, `运行auto.py` , 如果`mitmdump`运行界面不报错并正常返回数据, 说明运行成功, 抓取`url`和 `cookies`的数据会存在当前路径下

8. 当完成`url` 和 `cookies` 抓取后, 根据url 和 cookies 来获取评论数据, 首先在`config.py` 和 `crawl.py` 中找到`url_cookies_path` 和 `save_to` 修改一下文件读取和结果保存位置, 运行`python crawl.py` 进行抓取.



## 最后
如果有地方运行失败, 可以及时与作者沟通, 作者全天在线, 微信:1832803526, gmail:a1748270@student.adelaide.edu.au