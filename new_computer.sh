#!/bin/bash

apt-get -y update 
apt-get -y dist-upgrade
apt-get -y install gnome-session-fallback
#logout

#Stuff I dont want
apt-get -y purge nvidia-common
apt-get -y purge .*unity.*
apt-get -y install unity-greeter
apt-get -y install brasero brasero-cdrkit nautilus nautilus-sendto nautilus-share rhythmbox-plugin-cdrecorder shotwell 
apt-get -y purge .*ubuntuone.*
apt-get -y purge overlay-scrollbar
apt-get -y purge gnome-orca
apt-get -y purge onboard

#Stuff I need
apt-get -y install git-core keepassx pidgin chromium-browser vim ntp virtualbox meld gitg

#Rapcad dependencies
#apt-get -y install flex bison libcgal-dev libqt4-dev libdxflib-dev asciidoc source-highlight

#Sutff I want
#apt-get -y install qtcreator gcstar openssh-server picard transmission-gtk
#apt-get -y install libgnome2-0
