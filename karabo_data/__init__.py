# coding: utf-8
"""The karabo_data package.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

__version__ = "0.7.0"

from warnings import warn

from .reader import *
from .stacking import *
from .utils import *

def ZMQStreamer(port, maxlen=10, protocol_version='2.2', dummy_timestamps=False):
    warn("Please update imports: "
         "karabo_data.ZMQStreamer -> karabo_data.export.ZMQStreamer",
         UserWarning, stacklevel=2)
    from . import export
    return export.ZMQStreamer(
        port=port, maxlen=maxlen, protocol_version=protocol_version, dummy_timestamps=dummy_timestamps
    )

def serve_files(path, port, source_glob='*', key_glob='*', **kwargs):
    warn("Please update imports: "
         "karabo_data.serve_files -> karabo_data.export.serve_files",
         UserWarning, stacklevel=2)
    from . import export
    return export.serve_files(
        path=path, port=port, source_glob=source_glob, key_glob=key_glob, **kwargs
    )

__all__ = reader.__all__ + utils.__all__ + stacking.__all__ + ['ZMQStreamer', 'serve_files']
