import subprocess
from distutils.command.build import build as _build  # type: ignore

import setuptools

class build(_build):
  """
  A build command class that will be invoked during package install.
  The package built using the current setup.py will be staged and later
  installed in the worker using `pip install package'. This class will be
  instantiated during install for this specific scenario and will trigger
  running the custom commands specified.
  """

  sub_commands = _build.sub_commands + [('CustomCommands', None)]

CUSTOM_COMMANDS = [['echo', 'Custom command worked!']]


class CustomCommands(setuptools.Command):
  """
  A setuptools Command class able to run arbitrary commands.
  """

  def RunCustomCommand(self, command_list):
    print('Running command: %s' % command_list)
    p = subprocess.Popen(
        command_list,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    # Can use communicate(input='y\n'.encode()) if the command run requires
    # some confirmation.
    stdout_data, _ = p.communicate()
    print('Command output: %s' % stdout_data)
    if p.returncode != 0:
      raise RuntimeError(
          'Command %s failed: exit code: %s' % (command_list, p.returncode))

  def run(self):
    for command in CUSTOM_COMMANDS:
      self.RunCustomCommand(command)

REQUIRED_PACKAGES = [
    'tensorflow-cloud==0.1.15',
    'tfx[kfp]==0.30.1',
    'kfp-pipeline-spec<0.2.0,>=0.1.8',
    'google-api-python-client<2,>=1.7.8',
    'google-auth-httplib2==0.1.0',
    'google-auth-oauthlib==0.4.4'
]

setuptools.setup(
    name='Kubeflow NextBestOffer Model',
    version='0.1.0',
    description='NBO set workflow package.',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        'build': build,
        'CustomCommands': CustomCommands,
    })