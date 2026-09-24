# Third-party notice: SYD Dynamics EasyProfile SDK

The eight source files in this directory (all but `CMakeLists.txt` and this
notice) are the EasyProfile SDK from SYD Dynamics ApS, redistributed under the
BSD-2-Clause terms in each header (Copyright (c) 2017 SYD Dynamics ApS; the
`.cpp` files carry the copyright line and refer to their header). MARBLE uses
it to decode the IMU's serial protocol.

## Version and files

EasyProfile 1.1.9 (R), Jul 27 2016, per `EasyProfile.h`. The companion files
carry their own versions: `BasicTypes.h` V1.1.6 (R), `EasyObjectDictionary.h`
1.1.8 (R), `EasyProtocol.h` v2.1 (R), `EasyQueue.h` V1.0 (R). No header names
a repository URL, so there is no upstream commit to pin.

```
BasicTypes.h
EasyObjectDictionary.cpp   EasyObjectDictionary.h
EasyProfile.cpp            EasyProfile.h
EasyProtocol.cpp           EasyProtocol.h
EasyQueue.h
```

## Modifications

None by this project. Keep the files byte-exact and do not normalise line endings:
six use CRLF, `EasyProtocol.h` and `EasyQueue.h` use LF. `.gitattributes` marks them `-text`.

`CMakeLists.txt` is this project's. [`../imu.hpp`](../imu.hpp) adapts
code from the SDK example `main_example.cpp` (the `OnSerialRX` receive and
dispatch code and the unit comments in `parse_combo`) under the same terms. Its
header carries that example's notice, `COPYRIGHT(c) 2024 SYD Dynamics ApS`.

## Licence

The five headers carry this text, which covers all eight files:

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Copyright (c) 2017 SYD Dynamics ApS.

The build compiles these files into `libEasyProfile.so`, linked into
`../imu_nanobind.abi3.so`. Clause 2 applies to any binary distribution of
either: ship this notice with it.
