css = '''
<style>
.chat-message {
    padding: 1rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex; 
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b313e;
    text-align: right;
    border-radius: 10px;
    padding: 10px;
}
.chat-message.bot {
    background-color: #475063;
    text-align: left;
    border-radius: 10px;
    padding: 10px;
}
.chat-message .message {
    color: #fff;
    font-size: 16px;
}
.chat-label {
    font-weight: bold;
    color: #fff;
    margin-bottom: 5px;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="chat-label">Reply</div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="chat-label">You</div>
    <div class="message">{{MSG}}</div>
</div>
'''
